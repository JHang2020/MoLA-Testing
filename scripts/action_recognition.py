import logging
import os
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from dataset import BABEL_AR, BABEL_AR_120
from models.mola import ClipModel
import json
log = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库  将输出打印为列表
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # p: predict, t: GT
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()  # init a table for print
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # for each class
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):  # plot confusion matrix
        cm = self.matrix
        np.save('confusion_matrix.npy',cm)
        



with open('data/BABEL/label2action.json') as f:
    action_name = json.load(f)
action2_name = {}
for key, value in action_name.items():
    action2_name[value] = key
action_name = action2_name
@hydra.main(version_base=None, config_name="babel_ar_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    if cfg.dataset.dataset_name == 'BABEL-60':
        test_dataloader = prepare_test_dataset_babel_60(cfg)
    elif cfg.dataset.dataset_name == 'BABEL-120':
        test_dataloader = prepare_test_dataset_babel_120(cfg)
    else:
        raise NotImplementedError
    model, tokenizer = prepare_test_model(saved_cfg)
    eval_babel(saved_cfg, test_dataloader, model, tokenizer, part_enhanced=False)


def prepare_test_dataset_babel_60(cfg):
    test_dataset = BABEL_AR(
        cfg,
        eval_mode=True,
        patch_size=cfg.train.patch_size,
        fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )
    return test_dataloader

def prepare_test_dataset_babel_120(cfg):
    test_dataset = BABEL_AR_120(
        cfg,
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
    
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer


def eval_babel(cfg, test_dataloader, model, tokenizer=None, verbose=True, part_enhanced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 0
    all_label_texts = []
    correct_num = 0
    correct_num_k = 0

    with torch.no_grad():
        model.eval()

        test_pbar = tqdm(test_dataloader, leave=False)
        avg_bloss, avg_ploss = 0.0, 0.0
        motion_list = []
        label_list = []
        for batch in test_pbar:
            motions, labels = batch #labels: B 10
            motions = motions.to(device)
            step += len(motions)
            motion_embeddings = model.encode_motion(motions)[0]
            motion_embeddings = motion_embeddings / motion_embeddings.norm(dim=-1, keepdim=True)
            motion_list.append(motion_embeddings.cpu().numpy())
            label_list.append(np.array(labels).transpose(1,0))
        
        all_label = np.unique(np.concatenate(label_list,0).reshape(-1))
        ori_all_label = all_label.copy()
        print('total label number:', len(all_label))
        for i in range(len(all_label)):
            all_label[i] = ('A person ' + all_label[i]+'.')
        print(all_label)
        all_texts_token = tokenizer(
                list(all_label), padding=True, return_tensors="pt"
            ).to(device)
        all_text_embeddings = model.encode_text(all_texts_token)
        all_text_embeddings = all_text_embeddings / all_text_embeddings.norm(dim=-1, keepdim=True)
        all_text_embeddings = all_text_embeddings.cpu().numpy()
        all_motion_embeddings = np.concatenate(motion_list, axis=0)

        sim_mat = all_motion_embeddings @ all_text_embeddings.T
        pred_label = sim_mat.argmax(-1)
        pred_label_top5 = np.argsort(sim_mat, axis=-1)[:,::-1]
        target_label = np.concatenate(label_list,0)#N 10 (multi-label classification following the TMR++)
        for i, p in enumerate(pred_label):
            if ori_all_label[p] in target_label[i]:
                correct_num += 1
        for i, pred_list in enumerate(pred_label_top5):
            pred_list = pred_list[:5]
            for p in pred_list:
                if ori_all_label[p] in target_label[i]:
                    correct_num_k += 1
                    break

    print('Acc_top1: ', correct_num/step * 100, 'Acc_top5: ', correct_num_k/step * 100)
    return correct_num/step * 100


if __name__ == "__main__":
    main()
