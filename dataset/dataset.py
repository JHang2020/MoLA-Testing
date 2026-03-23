import codecs as cs
import random
from os.path import join as pjoin
import os
import cv2
import numpy as np
import torch.nn.functional as F

import torch
from einops import rearrange
from torch.utils import data
from tqdm import tqdm
import math
from math import sin, cos

PART_NUM = 5

def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    #TVC
    data_numpy = np.dot(data_numpy, R)
    return data_numpy

def random_rotate(seq):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)
        return temp

    # TVC->TVMC
    new_seq = seq.copy()[...,None,:]
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    return new_seq[...,0,:]

class TextMotionPartDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        patch_size=16,
        fps=None,
        hand=False,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.patch_size = patch_size
        self.fps = fps

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.keyids = id_list

        new_name_list = []
        length_list = []
        self.part_text_dir = 'data/{}/part_texts'.format(cfg.dataset.dataset_name)
        self.part_caption = {}


        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(cfg.dataset.motion_dir, name + ".npy"))[:,:22]
                if len(motion.shape) != 3:
                    continue
                if np.isnan(motion).any():
                    continue
                if motion.shape[0] == 0:
                    continue
                text_data = []
                flag = False
                used_name = []
                with cs.open(pjoin(cfg.dataset.text_dir, name + ".txt")) as f:
                    for index, line in enumerate(f.readlines()):
                        if eval_mode and index >= 1:
                            continue
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[
                                int(f_tag * cfg.dataset.fps) : int(
                                    to_tag * cfg.dataset.fps
                                )
                            ]
                            
                            if len(n_motion) == 0:
                                continue

                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": n_motion,
                                "length": len(n_motion),
                                "text": [text_dict],
                            }
                            used_name.append(new_name)
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                
                with cs.open(os.path.join(self.part_text_dir, name + '.txt')) as ff:
                    part_text = ff.readlines()
                    #The following code is to select the parts to be supervised 
                    #i.e., r/l arm, torso, r/l leg
                    if len(part_text) == 9: #no mirror text
                        self.part_caption[name] = part_text[3:8].copy()
                        self.part_caption[name][2] = part_text[-1]
                    elif len(part_text) == 18: #with mirro text
                        self.part_caption[name] = part_text[3:8].copy()
                        self.part_caption[name][2] = part_text[8]
                        self.part_caption[name] += part_text[12:17].copy()
                        self.part_caption[name][7] = part_text[17]
                    else:
                        print('Oops! Wrong part text in one item...')
                        self.part_caption[name] = ['The right arm is undefined.'] * 5
                        self.part_caption[name][2] = ['The torso is undefined.']
                    if len(used_name) != 0:
                        for n in used_name:
                            self.part_caption[n] = self.part_caption[name]
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )
        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list


        if self.cfg.dataset.dataset_name == "KIT-ML":
            self.kinematic_chain = [
                [3, 8, 9, 10],#rhand
                [3, 5, 6, 7],#lhand
                [0, 1, 2, 3, 4],#torso
                [0, 16, 17, 18, 19, 20],#rleg
                [0, 11, 12, 13, 14, 15],#lleg
            ]
        else:
            self.kinematic_chain = [
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
                [0, 3, 6, 9, 12, 15],
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
            ]

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]
            if self.cfg.dataset.dataset_name == "KIT-ML" and self.fps is not None:
                motion = self._subsample_to_20fps(motion, self.cfg.dataset.fps)
            use_mean_persample = False
            if use_mean_persample:
                mean = motion.mean(0)
                std = motion.std(0)
            else:
                mean, std = self.mean, self.std

            motion = (motion - mean[np.newaxis, ...]) / std[np.newaxis, ...]

            motion = self.use_kinematic(motion)

            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]
        self.cnt = 0

    def real_len(self):
        return len(self.data_dict)

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

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        part_cap = self.part_caption[self.name_list[idx]]
        if not self.eval_mode and len(part_cap) > PART_NUM:
            assert len(part_cap) == 10
            if random.random() > 0.5:
                part_cap = part_cap[:PART_NUM]
            else:
                part_cap = part_cap[PART_NUM:2 * PART_NUM]
        if self.eval_mode:
            part_cap = part_cap[:PART_NUM]
        part_mask = torch.zeros(PART_NUM)
        for i in range(PART_NUM):
            if 'undefined.' in part_cap[i]:
                part_mask[i] = 1
            if not isinstance(part_cap[i], str):
                part_cap[i] = part_cap[i][0]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]
        # Randomly select a caption
        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        if (not self.eval_mode) and self.cfg.preprocess.shear:
            if random.random() < 0.5:
                motion = shear(motion)
                
        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                C = motion.shape[2]
                padding_zeros = np.zeros((padding_len, D, C), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        motion = torch.tensor(motion).float().detach()
        motion = rearrange(motion, "t j c -> c t j")
        return caption, motion, part_cap, part_mask, m_length, item


class TextMotionPartEventDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        patch_size=16,
        fps=None,
        hand=False,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.patch_size = patch_size
        self.fps = fps

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.keyids = id_list
        new_name_list = []
        length_list = []
        self.part_text_dir = 'data/{}/part_texts'.format(cfg.dataset.dataset_name)
        self.part_caption = {}
        self.stage_text_dir = 'data/{}/event_texts'.format(cfg.dataset.dataset_name)
        self.stage_caption = {}

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(cfg.dataset.motion_dir, name + ".npy"))[:,:22]
                if len(motion.shape) != 3:
                    continue
                if np.isnan(motion).any():
                    continue
                if len(motion) == 0:
                    continue
                text_data = []
                flag = False
                used_name = []
                with cs.open(pjoin(cfg.dataset.text_dir, name + ".txt")) as f:
                    for index, line in enumerate(f.readlines()):
                        if eval_mode and index >= 1:
                            continue
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[
                                int(f_tag * cfg.dataset.fps) : int(
                                    to_tag * cfg.dataset.fps
                                )
                            ]
                            if len(n_motion) == 0:
                                continue

                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": n_motion,
                                "length": len(n_motion),
                                "text": [text_dict],
                            }
                            used_name.append(new_name)
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                
                with cs.open(os.path.join(self.stage_text_dir, name + '.txt')) as ff: 
                    stage_text = ff.readlines()
                    for il in range(len(stage_text)):
                        st = stage_text[il].index(']')
                        stage_text[il] = stage_text[il][st+1:].strip()
                    assert len(stage_text) >= 1
                    self.stage_caption[name] = stage_text.copy()
                    if len(used_name) != 0:
                        for n in used_name:
                            self.stage_caption[n] = self.stage_caption[name]


                with cs.open(os.path.join(self.part_text_dir, name + '.txt')) as ff:
                    part_text = ff.readlines()
                    if len(part_text) == 9:
                        self.part_caption[name] = part_text[3:8].copy()
                        self.part_caption[name][2] = part_text[-1]
                    elif len(part_text) == 18:
                        self.part_caption[name] = part_text[3:8].copy()
                        self.part_caption[name][2] = part_text[8]
                        self.part_caption[name] += part_text[12:17].copy()
                        self.part_caption[name][7] = part_text[17]
                        #print(self.part_caption[name])
                    else:
                        self.part_caption[name] = ['The right arm is relaxing by the side of the body, undefined.']*5
                        self.part_caption[name][2] = ['The torso is stationary, undefined.']
                    if len(used_name) != 0:
                        for n in used_name:
                            self.part_caption[n] = self.part_caption[name]
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )
        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

        if self.cfg.dataset.dataset_name == "KIT-ML":
            self.kinematic_chain = [
                [3, 8, 9, 10],#rhand
                [3, 5, 6, 7],#lhand
                [0, 1, 2, 3, 4],#torso
                [0, 16, 17, 18, 19, 20],#rleg
                [0, 11, 12, 13, 14, 15],#lleg
            ]
        else:
            self.kinematic_chain = [
                [9, 14, 17, 19, 21],
                [9, 13, 16, 18, 20],
                [0, 3, 6, 9, 12, 15],
                [0, 2, 5, 8, 11],
                [0, 1, 4, 7, 10],
            ]

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]
            if self.cfg.dataset.dataset_name == "KIT-ML" and self.fps is not None:
                motion = self._subsample_to_20fps(motion, self.cfg.dataset.fps)
            use_mean_persample = False
            if use_mean_persample:
                mean = motion.mean(0)
                std = motion.std(0)
            else:
                mean, std = self.mean, self.std

            motion = (motion - mean[np.newaxis, ...]) / std[np.newaxis, ...]

            motion = self.use_kinematic(motion)

            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]
        self.cnt = 0

    def real_len(self):
        return len(self.data_dict)

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

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        part_cap = self.part_caption[self.name_list[idx]]
        stage_cap = self.stage_caption[self.name_list[idx]]
        stage_mask = torch.zeros(4)
        while len(stage_cap) < 4:
            stage_mask[len(stage_cap)] = 1
            stage_cap.append('undefined')
            
            
        if not self.eval_mode and len(part_cap) > 5:
            assert len(part_cap) == 10
            if random.random() > 0.5:
                part_cap = part_cap[:5]
            else:
                part_cap = part_cap[5:10]
        if self.eval_mode:
            part_cap = part_cap[:5]
        part_mask = torch.zeros(5)
        for i in range(5):
            if 'undefined.' in part_cap[i]:
                part_mask[i] = 1
            if not isinstance(part_cap[i], str):
                part_cap[i] = part_cap[i][0]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]
        # Randomly select a caption
        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                C = motion.shape[2]
                padding_zeros = np.zeros((padding_len, D, C), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        motion = torch.tensor(motion).float().detach()
        motion = rearrange(motion, "t j c -> c t j")
        return caption, motion, part_cap, part_mask, stage_cap, stage_mask, m_length, item
