import logging
import os
import sys
from os.path import join as pjoin
from thop import profile

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
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    test_dataloader = prepare_test_dataset_part(saved_cfg)
    model, tokenizer = prepare_test_model(saved_cfg)
    
    train_dataloader = prepare_train_dataset(saved_cfg)

    # part_enhanced params: False: MoLA; True: MoLA++ 
    num_runs = 3
    all_metrics = []
    for run_idx in range(num_runs):
        print(f"========== Test Run {run_idx + 1}/{num_runs} ==========")
        metrics = eval_part(
            saved_cfg,
            train_dataloader,
            test_dataloader,
            model,
            tokenizer,
            part_enhanced=True,
        )
        all_metrics.append(metrics)
        print_metrics(metrics, prefix=f"Run {run_idx + 1}")

    avg_metrics = {
        key: float(np.mean([metrics[key] for metrics in all_metrics]))
        for key in all_metrics[0]
    }
    print("========== Average Test Result ==========")
    print_metrics(avg_metrics, prefix=f"Average of {num_runs} runs")


def print_metrics(metrics, prefix="Test"):
    print(f"{prefix} results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.5f}%")
    
def prepare_test_dataset_part(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    else:
        test_split_file = pjoin(cfg.dataset.data_root, "test.txt")
    print(test_split_file, cfg.dataset.data_root)
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
        test_dataset, batch_size=16, shuffle=False, num_workers=16
    )
    return test_dataloader

def prepare_train_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_alias, TOKENIZERS_PARALLELISM=True
        )
    except: # your local path
        tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/netdisk/zhangjh/Code/TMR/distill-bert/distill-bert', TOKENIZERS_PARALLELISM=True
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

def eval_part(cfg, train_dataloader, test_dataloader, model, tokenizer=None, verbose=True, part_enhanced=False,truncation=True,max_length=100,):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []
    all_part_tfeat = []
    all_part_mfeat = []
    part_masks = []
    name_list = []
    all_len_list = []
    step = 0
    cnt_len = 0
    attn_list = []
    with torch.no_grad():
        model.eval()
        train_bar = tqdm(train_dataloader, leave=False)
        train_text_feat = []
        train_motion_feat = []
        for batch in train_bar:
            try:
                texts, motions, _, _, m_length, _ = batch
            except:
                texts, motions, _, _, _, _, m_length, _ = batch
            texts = list(texts)
            motions = motions.to(device)
            texts_token = tokenizer(
                texts, padding=True, return_tensors="pt",truncation=truncation,max_length=max_length
            ).to(device)
            motion_features = model.encode_motion(motions)[0]
            text_features = model.encode_text(texts_token)
            
            motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            train_text_feat.append(text_features)
            train_motion_feat.append(motion_features)

        train_text_feat = torch.concatenate(train_text_feat, dim=0).cpu().numpy()
        train_motion_feat = torch.concatenate(train_motion_feat, dim=0).cpu().numpy()


        test_pbar = tqdm(test_dataloader, leave=False)
        avg_bloss, avg_ploss = 0.0, 0.0
        for batch in test_pbar:
            step += 1
            texts, motions, part_text, part_mask, m_length, img_indexs = batch
            texts = list(texts)
            part_text = list(part_text)
            for i in range(5):
                part_text[i] = list(part_text[i])

            motions = motions.to(device)
            part_text = np.array(part_text).transpose(1,0).flatten().tolist()

            texts_token = tokenizer(
                texts, padding=True, return_tensors="pt",truncation=truncation,max_length=max_length
            ).to(device)
            part_texts_token = tokenizer(
                part_text, padding=True, return_tensors="pt",truncation=truncation,max_length=max_length
            ).to(device)
            all_len_list.append(m_length)
            attn_save = False

            motion_features, text_features, part_text_features, bloss, ploss = model(motions, texts_token, part_texts_token, part_mask.cuda(),length=m_length.cuda())
            if attn_save:
                attn = model.encode_motion(motions, length=m_length.cuda(), return_attn=True)[1]
                attn_list.append(attn.cpu().numpy())
            part_motion_features = torch.stack(motion_features[1:], dim=1)
            motion_features = motion_features[0]

            avg_bloss += bloss.item()
            avg_ploss += ploss.item()
            part_masks.append(part_mask)
            for i in range(motion_features.size(0)):
                
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())
                all_part_tfeat.append(part_text_features[i].cpu().numpy())
                all_part_mfeat.append(part_motion_features[i].cpu().numpy())
                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i].item())
                name_list.append(test_dataloader.dataset.name_list[img_indexs[i].item()])
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
    

    if attn_save:
        attn_list = np.concatenate(attn_list, axis=0)
    # match test queries to target motions, get nearest neighbors
    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)
    train_t_test_m = 100 * train_text_feat.dot(all_imgs_feat.T)
    train_m_test_t = 100 * train_motion_feat.dot(all_captions_feat.T)
    all_img_idxs = np.array(all_img_idxs)

    part_sim_scores = np.zeros_like(sims_t2m)
    if part_enhanced:
        for i in range(5):
            part_mfeat = all_part_mfeat[:, i]
            part_tfeat = all_part_tfeat[:, i]
            part_sim = 100 * part_tfeat.dot(part_mfeat.T)
            part_sim[part_masks[:,i].astype(bool),:] = 0
            part_sim[:, part_masks[:,i].astype(bool)] = 0
            sims_t2m += 0.1 * part_sim
            part_sim_scores += 0.1 * part_sim
    
    sims_m2t = sims_t2m.T
    #################################
    # Toolbox
    def get_retrieved_videos(sims, k):
        argm = np.argsort(-sims, axis=1)
        topk = argm[:,:k].reshape(-1)
        retrieved_videos = np.unique(topk)
        return retrieved_videos

    # Returns list of indices to normalize from sims based on videos
    def get_index_to_normalize(sims, videos):
        argm = np.argsort(-sims, axis=1)[:,0]
        result = np.array(list(map(lambda x: x in videos, argm)))
        result = np.nonzero(result)
        return result

    def qb_norm(train_test, test_test):
        k = 1 
        beta = 10 
        retrieved_videos = get_retrieved_videos(train_test, k)
        test_test_normalized = test_test
        train_test = np.exp(train_test*beta)
        test_test = np.exp(test_test*beta)
        
        normalizing_sum = np.sum(train_test, axis=0)
        index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
        test_test_normalized[index_for_normalizing, :] = \
            np.divide(test_test[index_for_normalizing, :], normalizing_sum)
        return test_test_normalized
    #########################

    
    sims_t2m = qb_norm(train_t_test_m/100, sims_t2m/100) * 100

    metrics = {}
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        metrics[f"R@{k}"] = r
        if verbose:
            log.info(f"跨模态运动-文本描述匹配Top-{k} 结果召回率 R@{k}: {r:.2f}")

    metrics["跨模态运动-文本描述匹配中位数排名指标算术平均值"] = (np.median(ranks) + 1) / len(sims_t2m) * 100
    if verbose:
        log.info(f"跨模态运动-文本描述匹配中位数排名指标: {metrics['跨模态运动-文本描述匹配中位数排名指标算术平均值']:.5f}%")

    return metrics

if __name__ == "__main__":
    main()
