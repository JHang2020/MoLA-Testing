"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import logging
import os
import sys
from os.path import join as pjoin
from metrics import *
import yaml
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from dataset import TextMotionPartDataset
from models.mola import ClipModel

log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    model, tokenizer = prepare_test_model(saved_cfg)
    part_enhanced = cfg.eval.part_enhanced
    eval_part(saved_cfg, model, tokenizer, part_enhanced=part_enhanced)


def prepare_test_dataset(cfg, split='test'):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    else:
        test_split_file = pjoin(cfg.dataset.data_root, split+'.txt')
    test_dataset = TextMotionPartDataset(
        cfg,
        mean,
        std,
        test_split_file,
        eval_mode=True,
        patch_size=cfg.train.patch_size,
        fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )
    return test_dataloader


def prepare_test_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = cfg.model.motion_encoder
    text_encoder_alias = cfg.model.text_encoder
    motion_embedding_dims: int = 768
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_alias, TOKENIZERS_PARALLELISM=True
    )

    model = ClipModel(
        motion_encoder_alias=motion_encoder_alias,
        text_encoder_alias=text_encoder_alias,
        motion_embedding_dims=motion_embedding_dims,
        text_embedding_dims=text_embedding_dims,
        projection_dims=projection_dims,
        patch_size=cfg.train.patch_size,
        part_contrast=cfg.train.part_weight,
    )

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer


def eval_part(cfg, model, tokenizer=None, verbose=True, part_enhanced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    def compute_sim_matrix(test_dataloader, small_batch=False):
        all_imgs_feat = []
        all_captions_feat = []

        all_img_idxs = []
        all_captions = []
        all_part_tfeat = []
        all_part_mfeat = []
        part_masks = []
        step = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            avg_bloss, avg_ploss = 0.0, 0.0
            for batch in test_pbar:
                step += 1
                texts, motions, part_text, part_mask, m_length, img_indexs = batch
                motions = motions.to(device)
                part_text = np.array(part_text).transpose(1,0).flatten().tolist()

                texts_token = tokenizer(
                    texts, padding=True, return_tensors="pt"
                ).to(device)
                part_texts_token = tokenizer(
                    part_text, padding=True, return_tensors="pt"
                ).to(device)

                motion_features, text_features, part_text_features, bloss, ploss = model(motions, texts_token, part_texts_token, part_mask.cuda())
                part_motion_features = torch.stack(motion_features[1:], dim=1)
                motion_features = motion_features[0]

                avg_bloss += bloss
                avg_ploss += ploss
                part_masks.append(part_mask)
                for i in range(motion_features.size(0)):
                    all_imgs_feat.append(motion_features[i].cpu().numpy())
                    all_captions_feat.append(text_features[i].cpu().numpy())
                    all_part_tfeat.append(part_text_features[i].cpu().numpy())
                    all_part_mfeat.append(part_motion_features[i].cpu().numpy())
                    all_captions.append(texts[i])
                    all_img_idxs.append(img_indexs[i].item())
            avg_bloss /= len(test_dataloader)
            avg_ploss /= len(test_dataloader)

        all_captions = np.array(all_captions)
        for img_idx, caption in zip(all_img_idxs, all_captions):
            dataset_pair[img_idx] = np.where(all_captions == caption)[0]

        all_imgs_feat = np.vstack(all_imgs_feat)
        all_captions_feat = np.vstack(all_captions_feat)
        all_part_mfeat = np.stack(all_part_mfeat, axis=0)
        all_part_tfeat = np.stack(all_part_tfeat, axis=0)
        part_masks = torch.cat(part_masks, dim=0).cpu().numpy()
        
        # match test queries to target motions, get nearest neighbors
        sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)
        if part_enhanced:
            for i in range(5):
                part_mfeat = all_part_mfeat[:, i]
                part_tfeat = all_part_tfeat[:, i]
                part_sim = 100 * part_tfeat.dot(part_mfeat.T)
                part_sim[part_masks[:,i].astype(bool),:] = 0
                part_sim[:, part_masks[:,i].astype(bool)] = 0
                sims_t2m += 0.1 * part_sim
        
        if small_batch:
            N = len(all_imgs_feat)
            idx = np.arange(N)
            np.random.seed(0)
            np.random.shuffle(idx)
            idx_batches = [
                idx[32 * i : 32 * (i + 1)] for i in range(N // 32)
            ]
            sims_t2m_lis = []
            for indices in idx_batches:
                sims_t2m_lis.append(sims_t2m[indices][:, indices])
            sims_t2m = sims_t2m_lis
                
        return sims_t2m
    
    datasets = {}
    results = {}
    for protocol in ["normal", "nsim", "guo"]:
        # Load the dataset if not already
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                dataset = prepare_test_dataset(cfg)
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
            elif protocol == "nsim":
                datasets[protocol] = prepare_test_dataset(cfg, split='nsim_test')
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol in ["normal", "threshold"]:
                res = compute_sim_matrix(datasets[protocol])
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "nsim":
                res = compute_sim_matrix(datasets[protocol])
                results[protocol] = res
            elif protocol == "guo":
                results["guo"] = compute_sim_matrix(datasets[protocol], small_batch=True)
        result = results[protocol]

        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x
                metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
            protocol_name = protocol
        else:
            sim_matrix = result

            protocol_name = protocol
            if protocol == "threshold":
                emb = result["sent_emb"]
                threshold = 0.95
                protocol_name = protocol + f"_{threshold}"
            else:
                emb, threshold = None, None
            metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)

        print_latex_metrics(metrics)
        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(cfg.checkpoints_dir, metric_name)
        save_metric(path, metrics)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


if __name__ == "__main__":
    main()
