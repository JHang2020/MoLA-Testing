import logging
import os
import random
import sys
sys.path.insert(0, os.getcwd())

from os.path import join as pjoin
import codecs as cs
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from einops import rearrange
from models.m2t_model import MotionCaptionModel
#os.environ['MOVERSCORE_MODEL'] = '/path/to/distill-bert/'

from test_caption import evaluation_m2t
os.environ["TOKENIZERS_PARALLELISM"] = "true"
log = logging.getLogger(__name__)
from torch.utils.data._utils.collate import default_collate
from torch.cuda.amp.grad_scaler import GradScaler

scaler = GradScaler()  # 自动调整 loss scale，防止 float16 溢出

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training motion-2-text generative model'''
class Motion2TextDataset(Dataset):
    def __init__(self, dataset_name, split, w_vectorizer, max_text_len = 20):

        self.max_length = 224 #288
        self.max_motion_length = self.max_length
        self.pointer = 0
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.w_vectorizer = w_vectorizer
        self.eval_model = split=='test'
        self.patch_size = 16
        if dataset_name == 'HumanML3D':
            self.data_root = 'data/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joints')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            fps = 20
            self.guo_mean = np.load('data/HumanML3D/mean.npy')
            self.guo_std = np.load('data/HumanML3D/std.npy')
            self.kinematic_chain = [
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
                [0, 3, 6, 9, 12, 15],
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
            ]
        elif dataset_name == 'KIT-ML':
            self.data_root = './data/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joints')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            fps = 12.5
            self.kinematic_chain = [
                [3, 8, 9, 10],#rhand
                [3, 5, 6, 7],#lhand
                [0, 1, 2, 3, 4],#torso
                [0, 16, 17, 18, 19, 20],#rleg
                [0, 11, 12, 13, 14, 15],#lleg
            ]

        mean = np.load(f'data/{dataset_name}/Mean_raw.npy')
        std = np.load(f'data/{dataset_name}/Std_raw.npy')

        split_file = pjoin(self.data_root, f'{split}.txt')

        min_motion_len = 40 if self.dataset_name == 'HumanML3D' else 24

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                guo_motion = np.load(pjoin(self.motion_dir, name + '.npy').replace('new_joints','new_joint_vecs'))
                if self.eval_model:
                    if (len(motion)) < min_motion_len or (len(motion) >= self.max_length):
                        continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                n_guo_motion = guo_motion[int(f_tag * fps): int(to_tag * fps)]
                                if self.eval_model:
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_length):
                                        continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict],
                                                       'guo_motion':n_guo_motion}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                       'guo_motion': guo_motion}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
                pass
        
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]
            if dataset_name == "KIT-ML" and self.fps is not None:
                motion = self._subsample_to_20fps(motion, fps)
            
            mean, std = self.mean, self.std

            motion = (motion - mean[np.newaxis, ...]) / std[np.newaxis, ...]

            motion = self.use_kinematic(motion)

            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert len(self.data_dict)

    def _subsample_to_20fps(self, orig_ft, orig_fps):
        T, n_j, _ = orig_ft.shape
        out_fps = 20.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps) % int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T / int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration * int(orig_fps)
                t_idxs += [
                    x + last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T
                ]
        else:
            t_idxs = np.arange(0, T, orig_fps / out_fps, dtype=int)

        ft = orig_ft[t_idxs, :, :]
        return ft

    def use_kinematic(self, motion):
        #motion: legnth, joint, 3
        if self.patch_size == 16: 
            motion_ = np.zeros(
                (motion.shape[0], len(self.kinematic_chain) * 16, motion.shape[2]),
                float,
            )
            for i_frames in range(motion.shape[0]):
                for i, kinematic_chain in enumerate(self.kinematic_chain):
                    if len(kinematic_chain) == 0:
                        joint_parts = np.zeros((1,16,3)).astype('float')
                    else:
                        joint_parts = motion[i_frames, kinematic_chain]
                        joint_parts = joint_parts.reshape(1, -1, 3)# 1, joint_num, 3
                        joint_parts = cv2.resize(
                            joint_parts, (16, 1), interpolation=cv2.INTER_LINEAR
                        )# 1, jointnum->16, 3
                    motion_[i_frames, 16 * i : 16 * (i + 1)] = joint_parts[0]

        else:
            raise NotImplementedError

        return motion_
    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['pre_motion'], data['length'], data['text']

        # Randomly select a caption
        if self.eval_model:
            text_data = random.choice(text_list)#text_list[0]
        else:
            text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.eval_model:
            all_captions = [' '.join(
                [token.split('/')[0].strip() for token in text_dic['tokens']]
            ) for text_dic in text_list]

            if len(all_captions) > 3:
                all_captions = all_captions[:3]
            elif len(all_captions) == 2:
                all_captions = all_captions + all_captions[0:1]
            elif len(all_captions) == 1:
                all_captions = all_captions * 3


            if len(tokens) < self.max_text_len:         # max_text_len = 20
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        guo_motion = data['guo_motion']
        guo_motion = (guo_motion - self.guo_mean) / self.guo_std
        
        #following the MG-MotionLLM
        self.unit_length = 4
        
        if self.eval_model:    
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
            guo_motion = guo_motion[idx:idx + m_length]

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_model
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            guo_motion = guo_motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            padding_len = max_motion_length - m_length
            D = motion.shape[1]
            C = motion.shape[2]
            padding_zeros = np.zeros((padding_len, D, C), dtype=np.float32)
            motion = np.concatenate((motion, padding_zeros), axis=0)
            
            D = guo_motion.shape[1]
            padding_zeros = np.zeros((padding_len, D), dtype=np.float32)
            guo_motion = np.concatenate((guo_motion, padding_zeros), axis=0)

        motion = torch.tensor(motion).float().detach()
        motion = rearrange(motion, "t j c -> c t j")

        if self.eval_model:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, all_captions, torch.tensor(guo_motion).float(), m_length
        else:
            return motion, caption, m_length, name

@hydra.main(version_base=None, config_name="m2t_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    cfg.log_path = os.path.join(cfg.checkpoints_dir, 'log.txt')
    if not os.path.exists(cfg.log_path):
        open(cfg.log_path, 'w').close()

    set_seed(cfg.train.seed)

    train_dataloader, val_dataloader = prepare_dataset(cfg)
    model, optimizer, scheduler = prepare_model(cfg, train_dataloader)

    train(
        cfg,
        train_dataloader,
        val_dataloader,
        model,
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


from scripts.utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('data/HumanML3D/glove', 'our_vab')


def prepare_dataset(cfg):
    val_dataloader = torch.utils.data.DataLoader(
        Motion2TextDataset(cfg.dataset.dataset_name, 'test', w_vectorizer),
        32,#不能改
        shuffle = True,
        num_workers=16,
        collate_fn=collate_fn,
        drop_last=True
    )

    #train_dataloader = torch.utils.data.DataLoader(
    #    Motion2TextDataset(cfg.dataset.dataset_name, 'train', w_vectorizer),
    #    cfg.train.batch_size,
    #    shuffle = True,
    #    num_workers=8,
    #    drop_last=True
    #)

    return val_dataloader, val_dataloader


def prepare_model(cfg, train_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 1. 加载 Motion Encoder（假设已定义）
    # -----------------------------

    from models.mola import ClipModel
    motion_encoder = ClipModel(
        part_contrast=0.5
    )

    # 如果有预训练权重
    if cfg.model.motion_encoder_ckpt:
        ckpt = torch.load(cfg.model.motion_encoder_ckpt, map_location='cpu')
        motion_encoder.load_state_dict(ckpt, strict=True)
        print(f"Loaded motion encoder from {cfg.model.motion_encoder_ckpt}")

    # -----------------------------
    # 2. 构建 Caption Model
    # -----------------------------
    model = MotionCaptionModel(
        motion_encoder=motion_encoder,
        num_reg_tokens=cfg.model.num_reg_tokens,
        projection_hidden_size=cfg.model.projection_hidden_size,
        lora_r=cfg.model.lora_r,
        use_lora=cfg.model.use_lora,
    )

    if cfg.resume!=None:
        state_dict = torch.load(cfg.resume)
        msg = model.load_state_dict(state_dict, strict=False)
        #print(msg)

    model.to(device)
    model.train()

    # -----------------------------
    # 3. 优化器设置
    # -----------------------------
    # 只训练：projection 层 + (LoRA 参数)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,  # 注意：这里用基础 lr
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * cfg.train.epoch
    )

    return model, optimizer, scheduler

def train(
    cfg,
    train_dataloader,
    val_dataloader,
    model,
    optimizer,
    scheduler,
):
    # Evaluator Setting
    if cfg.dataset.dataset_name == 'kit':
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm = 1.0
    save_every = 25
    eval_every = 25

    best_loss = 1e5
    best_r1 = 0
    best_ep = -1
    global_step = 0
    
    if cfg.resume:
        num_runs = 3
        r3_list = []
        rouge_list = []

        for run_idx in range(num_runs):
            print(f"========== Test Run {run_idx + 1}/{num_runs} ==========")

            (
                r_1,
                r_2,
                r_3,
                matching_score_pred,
                bleu1,
                bleu2,
                rouge,
                cider,
                bert_score,
                msg,
            ) = evaluation_m2t(
                val_dataloader,
                cfg,
                model,
                model.opt_tokenizer,
                w_vectorizer,
                eval_wrapper=eval_wrapper,
                instruction="a person is",
                max_new_tokens=40,
            )

            r3_list.append(float(r_3.item() if hasattr(r_3, "item") else r_3))
            rouge_list.append(float(rouge.item() if hasattr(rouge, "item") else rouge))

            print("测试结果：")
            print("跨模态运动文本描述生成语义一致性指标Top-3 R-Precision: ", r_3)
            print("跨模态运动文本描述生成词汇级匹配指标ROUGE-L: ", rouge)

        print("========== 三次测试平均结果 ==========")
        print("跨模态运动文本描述生成语义一致性指标Top-3 R-Precision: ", sum(r3_list) / num_runs)
        print("跨模态运动文本描述生成词汇级匹配指标ROUGE-L: ", sum(rouge_list) / num_runs)

        return




    for epoch in range(cfg.train.epoch):
        print(f"Running epoch {epoch}, best val loss: {best_loss}, best m2t R@1: {best_r1} at epoch {best_ep}")

        # === Training ===
        model.train()
        tr_loss = 0
        step = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            step += 1
            global_step += 1
            optimizer.zero_grad()

            motions, captions, lengths, names = batch  # motions: [B, C, T]
            motions = motions.to(device).float()

            # Forward
            outputs = model(motion_seq=motions, captions=captions)
            loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            #print(loss.item())

            if step % 20 == 0:
                with open(cfg.log_path, 'a+') as f:
                    f.write(f"[Epoch {epoch}] Step {step}, Loss: {loss.item():.4f}\n")

        tr_loss /= step

        # Save latest
        if (epoch+1) % save_every == 0 or epoch == cfg.train.epoch - 1:
            ckpt_path = os.path.join(cfg.checkpoints_dir, f"last_model_epoch{epoch}.pt")
            
            trainable_state_dict = {}
            
            # 保存投影层参数（保持完整名称）
            for name, param in model.named_parameters():
                if 'projection' in name and param.requires_grad:
                    trainable_state_dict[name] = param.data
                elif 'lora' in name and param.requires_grad:
                    trainable_state_dict[name] = param.data
                elif param.requires_grad:
                    print(name, 'saving...')
                    trainable_state_dict[name] = param.data
            
            torch.save(trainable_state_dict, ckpt_path)


        # === Validation ===
        if (epoch+1) % eval_every == 0:
            model.eval()
            r_1, r_2, r_3, matching_score_pred, bleu1, bleu4, rouge, cider, bert_score, msg = evaluation_m2t(
                val_dataloader,
                cfg,
                model,
                model.opt_tokenizer,
                w_vectorizer,
                eval_wrapper=eval_wrapper,
                instruction='a person is',
                max_new_tokens=40
            )
            with open(cfg.log_path, 'a+') as f:
                f.write(msg+'\n')
            model.train()

            if bleu1 > best_r1:
                best_r1 = bleu1
                best_ep = epoch
                best_ckpt = os.path.join(cfg.checkpoints_dir, "best_model.pt")
                torch.save(model.state_dict(), best_ckpt)


from argparse import Namespace
import re
from os.path import join as pjoin
from scripts.utils.evaluator_wrapper import EvaluatorModelWrapper

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag

def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'finest'
    opt.checkpoints_dir = './checkpoints/'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt


if __name__ == "__main__":
    main()