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

import numpy as np
import timm
import torch
import torch.nn.functional as F
import transformers
from torch import nn
import random

lhand_indices = [9, 13, 16, 18, 20]
rhand_indices = [9, 14, 17, 19, 21]
torso_indices = [0, 3, 6, 9, 12, 15]
lleg_indices = [0, 1, 4, 7, 10]
rleg_indices = [0, 2, 5, 8, 11]
PART_NUM = 5


def local_direc_loss(motion, motion_part, text, text_part, part_masks):
    loss_sum = 0.0
    for idx in range(PART_NUM):
        part_mask = part_masks[:,idx].bool()
        motion_l = motion - motion_part[idx]
        text_l = text - text_part[idx]
        sim = torch.einsum('nc,nc->n', [F.normalize(motion_l, dim=1), F.normalize(text_l, dim=1)])
        loss = 1. -  sim
        loss_sum += loss[~part_mask].mean()
    return loss_sum / PART_NUM

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        try:
            self.text_model = transformers.AutoModel.from_pretrained(model_name)
        except:
            self.text_model = transformers.AutoModel.from_pretrained('/mnt/netdisk/zhangjh/Code/TMR/distill-bert/distill-bert')

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]

    
class MotionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        patch_size=16,
        reg_tokens=5,
    ) -> None:
        super().__init__()

        self.reg_tokens = reg_tokens
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="token",
                img_size=(224, patch_size * PART_NUM),
            )
        except:
            pretrained_cfg = timm.models.create_model(model_name).default_cfg
            print(pretrained_cfg)
            #custom your local path of the pretrained ViT weights
            pretrained_cfg['file'] = '/mnt/netdisk/zhangjh/Code/MotionPatches/vit_models/vit_base_patch16_224_in21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'
            
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="token",
                img_size=(224, patch_size * PART_NUM),
                pretrained_cfg=pretrained_cfg
            )

        self.model.patch_size = patch_size
        self.model.num_prefix_tokens += reg_tokens
        self.model.num_reg_tokens = reg_tokens
        self.model.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, self.model.embed_dim)) if reg_tokens else None
        self.model.pos_embed = nn.Parameter(torch.cat([self.model.pos_embed[:,0:1,:].clone().repeat(1,self.reg_tokens,1), self.model.pos_embed], dim=1))
        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0
    
    def forward(self, x, length=None, patch_idx=None):
        '''
            obtain part tokens by global registered tokens
        '''
        # x.shape bs, 3, 224, 5*16
        x, feat = self.model.forward_intermediates(x)
        x = x[:,0:1+self.reg_tokens,:]
        x = self.model.fc_norm(x)
        x = self.model.head_drop(x)
        return x.unbind(1)
    
class ClipModel(nn.Module):
    def __init__(
        self,
        motion_encoder_alias="vit_base_patch16_224_in21k",
        text_encoder_alias="distilbert-base-uncased",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        logit: float = 0.07,
        patch_size: int = 16,
        part_contrast: float = -1.0
    ) -> None:
        super().__init__()

        motion_encoder = MotionEncoder(
            model_name=motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            patch_size=patch_size,
            reg_tokens= PART_NUM if part_contrast > 0 else 0,
        )
        self.patch_size = patch_size
        text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.part = (part_contrast > 0)
        self.part_weight = part_contrast
        if not self.part:
            self.motion_projection = ProjectionHead(
                embedding_dim=self.motion_encoder.model.embed_dim,
                projection_dim=projection_dims,
                dropout=dropout,
            )
        else:
            self.motion_projection = nn.ModuleList([ProjectionHead(
                embedding_dim=self.motion_encoder.model.embed_dim,
                projection_dim=projection_dims,
                dropout=dropout,
            ) for _ in range(PART_NUM + 1)])

        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode_motion(self, motion, length=None, patch_idx=None):
        if not self.part: #baseline
            motion_features = self.motion_encoder(motion, length=length)[0]
            motion_embeddings = self.motion_projection(motion_features)
        else:
            motion_features = self.motion_encoder(motion, length=length)
            motion_embeddings = []
            for i in range(PART_NUM + 1):
                motion_embeddings.append(self.motion_projection[i](motion_features[i]))
        return motion_embeddings

    def encode_text(self, text, pre_norm=False):
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )
        if pre_norm:
            return text_features
        text_embeddings = self.text_projection(text_features)

        return text_embeddings

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0
    def mix_part(self, motion, part_text_embed, part_mask, m_length):
        #motion: bs, 3, 224, 5*16
        #the motion order is rarm, larm, torso, rleg, lleg
        #the text order is rarm larm torso rleg lleg
        if random.random()<0.5:
            spa_part_idx = random.sample(list(range(PART_NUM)), k=2)
        else:
            spa_part_idx = random.sample(list(range(PART_NUM)), k=3)
        
        spa_part_idx.sort()
        spa_joint_idx = []
        for idx in spa_part_idx:
            spa_joint_idx += list(range(idx*self.patch_size, (idx+1)*self.patch_size))
        spa_joint_idx.sort()

        N = motion.shape[0]
        # generate swap swap idx
        idx = torch.arange(N)
        if N<=2:
            n = 1
        else:
            n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
        
        xst = motion.clone()
        for i in range(N):
            length_pattern = m_length[randidx[i]]
            length_ori = m_length[i]
            #resize pattern 1, 3, length_ori, 5*16
            if m_length[i] == 0 or m_length[randidx[i]] == 0:
                randidx[i] = i
                continue
            resize_pattern = torch.nn.functional.interpolate(motion[randidx[i], :, :length_pattern, :][None,...].transpose(2,3), (motion.shape[-1], length_ori), mode='bilinear', align_corners=False).transpose(2,3)
            xst[i,:, :length_ori, spa_joint_idx] = resize_pattern[0,:,:,spa_joint_idx].clone()
    
        part_text_embed[:, torch.tensor(spa_part_idx).cuda(), :] = part_text_embed[randidx][:, torch.tensor(spa_part_idx).cuda(), :]
        part_mask[:, spa_part_idx] = part_mask[randidx][:, spa_part_idx]

        lamb = len(spa_part_idx) / PART_NUM

        return xst, part_text_embed, part_mask, lamb, randidx

    def forward_mix_local_direction(self, motion, text, part_text=None, part_mask=None, return_loss=False, m_length=None):
        motion_embeds = self.encode_motion(motion, m_length)
        text_embeds = self.encode_text(text)
        if self.part:
            bs = motion_embeds[0].shape[0]
            part_text_embeds = self.encode_text(part_text).reshape(bs, PART_NUM, -1)
            mix_motion, mix_part_embeds, mix_part_mask, lamb, randidx = self.mix_part(motion, part_text_embeds, part_mask, m_length)
            mix_motion_embeds = self.encode_motion(mix_motion, m_length)

            for i in range(PART_NUM + 1):
                motion_embeds[i] = motion_embeds[i] / motion_embeds[i].norm(dim=-1, keepdim=True)
                mix_motion_embeds[i] = mix_motion_embeds[i] / mix_motion_embeds[i].norm(dim=-1, keepdim=True)
            part_text_embeds = part_text_embeds / part_text_embeds.norm(dim=-1, keepdim=True)
            mix_part_embeds = mix_part_embeds / mix_part_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            mix_text_embeds = text_embeds[randidx] * lamb + text_embeds * (1-lamb)
            mix_text_embeds = mix_text_embeds / mix_text_embeds.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds[0].t()) * logit_scale
            part_contrast_loss = 0.0
            mix_part_contrast_loss = 0.0
            for i in range(PART_NUM):
                if part_mask is not None:
                    pmask = part_mask[:, i].bool()
                    if pmask.long().sum() >= bs - 1:
                        continue
                else:
                    pmask = torch.zeros(bs).cuda()
                part_logits_per_text = torch.matmul(part_text_embeds[~pmask,i,:], motion_embeds[i+1][~pmask,:].t()) * logit_scale
                part_contrast_loss += self.clip_loss(part_logits_per_text)
            
            for i in range(PART_NUM):
                if mix_part_mask is not None:
                    pmask = mix_part_mask[:, i].bool()
                    if pmask.long().sum() >= bs - 1:
                        continue
                else:
                    pmask = torch.zeros(bs).cuda()
                mix_part_logits_per_text = torch.matmul(mix_part_embeds[~pmask,i,:], mix_motion_embeds[i+1][~pmask,:].t()) * logit_scale
                mix_part_contrast_loss += self.clip_loss(mix_part_logits_per_text)
            
            local_loss = local_direc_loss(motion_embeds[0], motion_embeds[1:], text_embeds, part_text_embeds.unbind(1), part_mask)
            if return_loss:
                return self.clip_loss(logits_per_text), self.clip_loss(torch.matmul(mix_text_embeds, mix_motion_embeds[0].t()) * logit_scale), part_contrast_loss / PART_NUM, mix_part_contrast_loss / PART_NUM, local_loss
            else:
                return motion_embeds, text_embeds, part_text_embeds, self.clip_loss(logits_per_text), part_contrast_loss / PART_NUM
        else:
            # normalized features
            motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale

            if return_loss:
                return self.clip_loss(logits_per_text)
            else:
                return motion_embeds, text_embeds

    def forward(self, motion, text, part_text=None, part_mask=None, return_loss=False,length=None):
        motion_embeds = self.encode_motion(motion,length=length)
        text_embeds = self.encode_text(text)
        if self.part:
            bs = motion_embeds[0].shape[0]
            part_text_embeds = self.encode_text(part_text).reshape(bs, PART_NUM, -1)
            for i in range(PART_NUM + 1):
                motion_embeds[i] = motion_embeds[i] / motion_embeds[i].norm(dim=-1, keepdim=True)
            part_text_embeds = part_text_embeds / part_text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds[0].t()) * logit_scale
            part_contrast_loss = 0.0
            for i in range(PART_NUM):
                if part_mask is not None:
                    pmask = part_mask[:, i].bool()
                    if pmask.long().sum() >= bs - 1:
                        return 0.0
                else:
                    pmask = torch.zeros(bs).cuda()
                part_logits_per_text = torch.matmul(part_text_embeds[~pmask,i,:], motion_embeds[i+1][~pmask,:].t()) * logit_scale
                part_contrast_loss += self.clip_loss(part_logits_per_text)
            
            if return_loss:
                return self.clip_loss(logits_per_text), part_contrast_loss / PART_NUM
            else:
                return motion_embeds, text_embeds, part_text_embeds, self.clip_loss(logits_per_text), part_contrast_loss / PART_NUM
        else:
            # normalized features
            motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale

            if return_loss:
                return self.clip_loss(logits_per_text)
            else:
                return motion_embeds, text_embeds
