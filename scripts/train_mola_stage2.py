import itertools
import logging
import os
import random
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from dataset import TextMotionPartEventDataset
from models.mola import ClipModel
from scripts.test import eval_part, prepare_test_dataset_part

os.environ["TOKENIZERS_PARALLELISM"] = "true"

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config_s2", config_path="../conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    cfg.log_path = os.path.join(cfg.checkpoints_dir, 'log.txt')
    if not os.path.exists(os.path.join(cfg.checkpoints_dir, 'log.txt')):
        file = open(cfg.log_path,'w')
        file.close()
    set_seed(cfg.train.seed)
    train_dataloader, test_dataloader = prepare_dataset(cfg)
    eval_dataloader = prepare_test_dataset_part(cfg)
    test_dataloader = eval_dataloader
    model, optimizer, scheduler, tokenizer = prepare_model(cfg, train_dataloader)
    train(
        cfg,
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
    )


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)



def prepare_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

    train_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    train_dataset = TextMotionPartEventDataset(
        cfg,
        mean,
        std,
        train_split_file,
        patch_size=cfg.train.patch_size,
        fps=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )

    return train_dataloader, train_dataloader


def prepare_model(cfg, train_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = cfg.model.motion_encoder
    text_encoder_alias = cfg.model.text_encoder
    motion_encoder_pretrained = cfg.train.motion_encoder_pretrained
    motion_encoder_trainable: bool = cfg.train.train_motion_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
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
        motion_encoder_alias,
        text_encoder_alias,
        motion_encoder_pretrained,
        motion_encoder_trainable,
        text_encoder_trainable,
        motion_embedding_dims,
        text_embedding_dims,
        projection_dims,
        patch_size=cfg.train.patch_size,
        dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
        part_contrast=cfg.train.part_weight,
    )

    model.to(device)
    if cfg.resume:
        print('Loading ', cfg.resume)
        state_dict = torch.load(cfg.resume)
        model.load_state_dict(state_dict)
    
    model.to(device)

    parameters = [
        {
            "params": model.motion_encoder.parameters(),
            "lr": cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor,
        },
        {
            "params": itertools.chain(
                model.motion_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor,
        },
    ]
    optimizer = optim.Adam(parameters)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader) * cfg.train.epoch * 2
    )

    return model, optimizer, scheduler, tokenizer


def train(
    cfg,
    train_dataloader,
    test_dataloader,
    eval_dataloader,
    model,
    tokenizer,
    optimizer,
    scheduler,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_h_r1 = 0
    best_ep = -1
    best_mean = 0
    for epoch in range(cfg.train.epoch):
        print(
            f"running epoch {epoch}, best test loss {best_te_loss} best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} best_mean {best_mean} after epoch {best_ep}"
        )
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            texts, motions, part_text, part_mask, stage_text, stage_mask, m_length, _ = batch
            part_text = np.array(part_text).transpose(1,0).flatten().tolist()
            stage_text = np.array(stage_text).transpose(1,0).flatten().tolist()

            motions = motions.to(device)
            m_length = m_length.to(device)

            texts = tokenizer(
                texts, padding=True, return_tensors="pt"
            ).to(device)
            part_text = tokenizer(
                part_text, padding=True, return_tensors="pt"
            ).to(device)
            stage_text = tokenizer(
                stage_text, padding=True, return_tensors="pt"
            ).to(device)

            
            contrast_loss, mix_contrast_loss, part_loss, mix_part_loss, stage_contrast_loss, local_loss, loss_order = model.forward_train(motions, texts, part_text, part_mask.cuda(), stage_text, stage_mask.cuda(), return_loss=True, m_length=m_length)
            loss_entropy = torch.zeros(1,)
            total_loss = contrast_loss + (cfg.train.part_weight * part_loss)
            total_loss = total_loss + (mix_contrast_loss + mix_part_loss * cfg.train.part_weight) * cfg.train.mix_weight
            total_loss = total_loss + (stage_contrast_loss * cfg.train.stage_weight)
            total_loss = total_loss + (local_loss * 0.1)
            total_loss = total_loss + (loss_order * cfg.train.order_weight)

            total_loss.backward()
            tr_loss += total_loss.item()
            optimizer.step()
            scheduler.step()
            if step % 20 ==0:
                with open(cfg.log_path, 'a+') as f:
                    f.write(f"train bodyCE: {contrast_loss.item()} stageCE: {stage_contrast_loss.item()} train mix bodyCE: {mix_contrast_loss.item()} mix partCE: {mix_part_loss.item()}  local_loss: {local_loss.item()} order_loss: {loss_order.item()} entropy_loss: {loss_entropy.item()} all: {total_loss.item()}"+'\n')
            pbar.set_description(f"train bodyCE: {contrast_loss.item()} stageCE: {stage_contrast_loss.item()} train mix bodyCE: {mix_contrast_loss.item()} mix partCE: {mix_part_loss.item()} local_loss: {local_loss.item()} order_loss: {loss_order.item()} entropy_loss: {loss_entropy.item()} all: {total_loss.item()}", refresh=True)
        tr_loss /= step

        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "last_model.pt"))

        t2m_r1, m2t_r1, avg_bloss, avg_ploss = eval_part(
            cfg, train_dataloader, eval_dataloader, model, tokenizer=tokenizer, verbose=False
        )

        log.info(
            f"epoch {epoch}, tr_loss {tr_loss}, vb_loss {avg_bloss}, vp_loss {avg_ploss}, t2m_r1 {t2m_r1}, m2t_r1 {m2t_r1}, best mean {max(best_mean, (t2m_r1+m2t_r1)/2.0)} "
        )
        with open(cfg.log_path, 'a+') as f:
            f.write("epoch {}, tr_loss {}, vb_loss {}, vp_loss {}, t2m_r1 {}, m2t_r1 {}".format(epoch, tr_loss, avg_bloss,avg_ploss, t2m_r1, m2t_r1)+'\n')
        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)
        best_mean = max(best_mean, (t2m_r1+m2t_r1)/2.0)
        if best_mean == (t2m_r1+m2t_r1)/2.0:
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))
        del total_loss, motions, texts
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
