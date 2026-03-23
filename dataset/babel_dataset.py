import codecs as cs
import random
from os.path import join as pjoin
import os
import cv2
import numpy as np
import pickle
import torch
from einops import rearrange
from torch.utils import data
from tqdm import tqdm
from dataset.convert_272_to_joint import recover_from_local_position

class BABEL_AR(data.Dataset):
    def __init__(
        self,
        cfg,
        eval_mode=False,
        patch_size=16,
        fps=None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.patch_size = patch_size
        self.fps = fps
        self.data_path = f'{cfg.dataset.data_root}/new_joints'
        print(self.data_path)
        f_read = open('data/BABEL/label60.pkl', 'rb')
        self.label = pickle.load(f_read)
        f_read.close()
        max_multilabel = 10 #最多一个sequence可以包含10个label, 不足则重复补齐
        for k in self.label:
            while len(self.label[k]) < max_multilabel:
                self.label[k].append(self.label[k][-1])
        
        with open('data/BABEL/test_no_mirror.txt', 'r') as f:
            self.data_items = list(f.readlines())
            idx_list = []
            for i in range(len(self.data_items)):
                self.data_items[i] = self.data_items[i].strip()
            
                if os.path.exists(os.path.join(self.data_path, self.data_items[i]+'.npy')):
                    idx_list.append(i)
            self.data_items = np.array(self.data_items)[np.array(idx_list)]
        self.get_mean_map()
        self.kinematic_chain = [
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
            [0, 3, 6, 9, 12, 15],
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
        ]

    def real_len(self):
        return len(self.data_items)

    def get_mean_map(self):
        #using mean/std from HumanML3D
        self.mean_map = np.load('data/HumanML3D/Mean_raw.npy')[None]
        self.std_map = np.load('data/HumanML3D/Std_raw.npy')[None]
        
        print(self.mean_map.mean(0).mean(0))

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
        return len(self.data_items)

    def __getitem__(self, item):
        index = item
        name = self.data_items[index]
        key = name
        label = self.label[key]
        try:
            data_numpy = np.load(os.path.join(self.data_path, name+'.npy'))
        except:
            print('error:', name)
            return self.__getitem__(0)
        
        if len(data_numpy.shape) != 3:
            print('error shape in', name)
            return self.__getitem__(0)
        
        motion = (data_numpy - self.mean_map) / self.std_map
        motion = self.use_kinematic(motion)# t j c
        m_length = motion.shape[0]

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
        
        motion = torch.tensor(motion).float()
        motion = rearrange(motion, "t j c -> c t j")
        return motion, label

class BABEL_AR_120(data.Dataset):
    def __init__(
        self,
        cfg,
        eval_mode=False,
        patch_size=16,
        fps=None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.patch_size = patch_size
        self.fps = fps
        self.data_path = f'{cfg.dataset.data_root}/new_joints'
        
        f_read = open('data/BABEL/label120.pkl', 'rb')
        self.label = pickle.load(f_read)
        f_read.close()
        max_multilabel = 10
        for k in self.label:
            while len(self.label[k]) < max_multilabel:
                self.label[k].append(self.label[k][-1])
        
        with open('data/BABEL/test_no_mirror_120.txt', 'r') as f:
            self.data_items = list(f.readlines())
            idx_list = []
            for i in range(len(self.data_items)):
                self.data_items[i] = self.data_items[i].strip()
            
                if os.path.exists(os.path.join(self.data_path, self.data_items[i]+'.npy')):
                    idx_list.append(i)
            self.data_items = np.array(self.data_items)[np.array(idx_list)]
        self.get_mean_map()
        self.kinematic_chain = [
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
            [0, 3, 6, 9, 12, 15],
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
        ]

    def real_len(self):
        return len(self.data_items)

    def get_mean_map(self):
        self.mean_map = np.load('data/HumanML3D/Mean_raw.npy')[None]#data.mean(axis=0, keepdims=True)# 1 V C 
        self.std_map = np.load('data/HumanML3D/Std_raw.npy')[None]#data.reshape((T, V*C)).std(axis=0).reshape((1, V, C))
        

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
        return len(self.data_items)

    def __getitem__(self, item):
        index = item
        name = self.data_items[index]
        key = name
        label = self.label[key]
        try:
            data_numpy = np.load(os.path.join(self.data_path, name+'.npy'))
        except:
            print('error:', name)
            return self.__getitem__(0)
        
        if len(data_numpy.shape) != 3:
            print('error shape in', name)
            return self.__getitem__(0)
        
        motion = (data_numpy - self.mean_map) / self.std_map
        motion = self.use_kinematic(motion)# t j c
        m_length = motion.shape[0]

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
        
        motion = torch.tensor(motion).float()
        motion = rearrange(motion, "t j c -> c t j")
        return motion, label