import numpy as np
import timm
import torch
import torch.nn.functional as F
import transformers
from torch import nn
import random


def generate_length_mask_2d(length):
    patch_size = 16
    length = (length + 15) // patch_size  # convert to patch count
    B = length.shape[0]
    token_length = 224 // patch_size  # 14
    idx = torch.arange(0, token_length, 1)[None].cuda().long().repeat(B, 1)  # B x N

    frame_num_expand = length.reshape(B, 1)  # B x 1
    p_mask = (idx < frame_num_expand).long().reshape(B, token_length)
    return p_mask  # B x P=14


def local_direc_loss(motion, motion_part, text, text_part, part_masks):
    """
    Compute directional consistency loss between global and part-level embeddings.

    For each part/event, measures the alignment between the relative direction of
    (global - part) in motion space and the same direction in text space.

    Args:
        motion:      Global motion embedding,       shape (N, D)
        motion_part: List of K part motion embeds,  each shape (N, D)
        text:        Global text embedding,         shape (N, D)
        text_part:   List of K part text embeds,    each shape (N, D)
        part_masks:  Boolean mask for missing parts, shape (N, K)

    Returns:
        Scalar loss averaged over all K parts.
    """
    num_parts = len(motion_part)
    loss_sum = 0.0
    for idx in range(num_parts):
        part_mask = part_masks[:, idx].bool()
        motion_l = motion - motion_part[idx]
        text_l = text - text_part[idx]
        sim = torch.einsum('nc,nc->n', [F.normalize(motion_l, dim=1), F.normalize(text_l, dim=1)])
        loss = 1. - sim
        loss_sum += loss[~part_mask].mean()
    return loss_sum / num_parts


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with residual connection and LayerNorm."""

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
        x += projected  # residual connection
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    """Wraps a HuggingFace transformer and extracts the [CLS] token representation."""

    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()

        try:
            self.text_model = transformers.AutoModel.from_pretrained(model_name)
        except:
            self.text_model = transformers.AutoModel.from_pretrained('/mnt/netdisk/zhangjh/Code/TMR/distill-bert/distill-bert')

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0  # [CLS] position

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class MotionEncoder(nn.Module):
    """
    ViT-based motion encoder that treats skeleton sequences as 2-D image patches.

    The spatial axis maps to body-part columns (patch_size px each) and the
    temporal axis maps to rows (224 px).  Optional register tokens are prepended
    to capture part-level representations alongside the global [CLS] token.

    Args:
        model_name:  timm model identifier (overridden to vit_base_patch16_224_in21k).
        pretrained:  Whether to load ImageNet-21k pretrained weights.
        trainable:   Whether encoder parameters are updated during training.
        patch_size:  Width of each body-part column in pixels (default 16).
        reg_tokens:  Number of register tokens appended after [CLS] (0 = disabled).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        patch_size: int = 16,
        reg_tokens: int = 5,
    ) -> None:
        super().__init__()

        self.reg_tokens = reg_tokens
        model_name = 'vit_base_patch16_224_in21k'
        pretrained_cfg = timm.models.create_model('vit_base_patch16_224_in21k').default_cfg
        
        #custom your local path of the pretrained ViT weights
        pretrained_cfg['file'] = '/mnt/netdisk/zhangjh/Code/MotionPatches/vit_models/vit_base_patch16_224_in21k/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'
            
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="token",
            img_size=(224, patch_size * 5),
            pretrained_cfg=pretrained_cfg,
        )
        self.model.patch_size = patch_size

        # Extend prefix tokens with register tokens
        self.model.num_prefix_tokens += reg_tokens
        self.model.num_reg_tokens = reg_tokens
        self.model.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, self.model.embed_dim))
            if reg_tokens else None
        )
        # Expand positional embeddings to account for register tokens
        self.model.pos_embed = nn.Parameter(
            torch.cat([
                self.model.pos_embed[:, 0:1, :].clone().repeat(1, self.reg_tokens, 1),
                self.model.pos_embed,
            ], dim=1)
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x, length=None, return_seq=None):
        """
        Args:
            x:          Input tensor of shape (B, 3, 224, part_num * patch_size).
            length:     Per-sample valid frame lengths, shape (B,).
            return_seq: If True, also return the per-frame sequence features.

        Returns:
            Tuple of token tensors (cls + reg_tokens) unpacked along dim 1.
            When return_seq=True, additionally returns sequence features (B, T, D).
        """
        x, feat = self.model.forward_intermediates(x)

        if return_seq:
            x = self.model.fc_norm(x)
            x = self.model.head_drop(x)
            # Reshape patch tokens into (B, T, num_parts, D) and average over parts
            seq_feat = x[:, 1 + self.reg_tokens:, :].reshape(
                x.shape[0], -1, 5, x.shape[-1]
            )
            seq_feat = seq_feat.mean(2)  # B x T x D
            return x[:, 0:1 + self.reg_tokens, :].unbind(1), seq_feat

        x = x[:, 0:1 + self.reg_tokens, :]
        x = self.model.fc_norm(x)
        x = self.model.head_drop(x)
        return x.unbind(1)


class ClipModel(nn.Module):
    """
    CLIP-style motion–text contrastive model.

    Supports two training modes:
      - Baseline: global motion vs. text contrastive loss only.
      - Part-aware (part_contrast > 0): adds per-body-part contrastive losses,
        a mixed-sample part loss, temporal stage ordering loss, and a local
        directional consistency loss.

    Args:
        motion_encoder_alias:     timm model name for the motion encoder.
        text_encoder_alias:       HuggingFace model name for the text encoder.
        motion_encoder_pretrained: Load pretrained motion encoder weights.
        motion_encoder_trainable:  Fine-tune motion encoder during training.
        text_encoder_trainable:    Fine-tune text encoder during training.
        motion_embedding_dims:    Output dimension of the motion encoder.
        text_embedding_dims:      Output dimension of the text encoder.
        projection_dims:          Shared embedding space dimension.
        dropout:                  Dropout rate inside projection heads.
        logit:                    Initial temperature (1 / logit_scale).
        patch_size:               Width of each body-part patch column.
        part_contrast:            Weight for part-level losses; ≤0 disables them.
    """

    def __init__(
        self,
        motion_encoder_alias: str = "vit_base_patch16_224_in21k",
        text_encoder_alias: str = "distilbert-base-uncased",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        logit: float = 0.07,
        patch_size: int = 16,
        part_contrast: float = -1.0,
    ) -> None:
        super().__init__()

        motion_encoder = MotionEncoder(
            model_name=motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            patch_size=patch_size,
            reg_tokens=5 if part_contrast > 0 else 0,
        )
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
            # One head for global [CLS] + one per body part
            self.motion_projection = nn.ModuleList([
                ProjectionHead(
                    embedding_dim=self.motion_encoder.model.embed_dim,
                    projection_dim=projection_dims,
                    dropout=dropout,
                )
                for _ in range(1 + 5)
            ])

        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.part_num = 5

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_motion(self, motion, length=None, return_seq=None):
        """
        Encode motion input and project to the shared embedding space.

        Returns a list of embeddings [global, part_0, ..., part_4] when
        part_contrast is enabled, or a single tensor otherwise.
        When return_seq=True, also returns per-frame sequence embeddings.
        """
        if not self.part:  # baseline: global token only
            motion_features = self.motion_encoder(motion, length=length)[0]
            motion_embeddings = self.motion_projection(motion_features)
        elif return_seq:
            motion_features, seq_features = self.motion_encoder(
                motion, length=length, return_seq=True
            )
            motion_embeddings = [
                self.motion_projection[i](motion_features[i])
                for i in range(1 + self.part_num)
            ]
            seq_features = self.motion_projection[0](seq_features)
            return motion_embeddings, seq_features
        else:
            motion_features = self.motion_encoder(motion, length=length)
            motion_embeddings = [
                self.motion_projection[i](motion_features[i])
                for i in range(1 + self.part_num)
            ]
        return motion_embeddings

    def encode_text(self, text, pre_norm: bool = False):
        """
        Encode tokenised text and project to the shared embedding space.

        Args:
            text:     Dict with 'input_ids' and 'attention_mask'.
            pre_norm: If True, return raw encoder features before projection.
        """
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )
        if pre_norm:
            return text_features
        return self.text_projection(text_features)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """Symmetric CLIP loss averaged over both directions."""
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0

    # ------------------------------------------------------------------
    # CutMix-style spatial part mixing
    # ------------------------------------------------------------------

    def mix_part(self, motion, part_text_embed, part_mask, m_length):
        """
        Randomly swap 2–3 body-part columns between samples in the batch,
        producing mixed motion sequences and correspondingly mixed text embeddings.

        Args:
            motion:          (B, 3, 224, part_num * patch_size)
            part_text_embed: (B, part_num, D)
            part_mask:       (B, part_num) — 1 where the part annotation is missing
            m_length:        (B,) valid frame lengths

        Returns:
            xst:            Mixed motion tensor
            part_text_embed: Mixed part text embeddings
            part_mask:      Mixed part mask
            lamb:           Mixing ratio (fraction of parts swapped)
            randidx:        Index tensor mapping each sample to its swap source
        """
        # Randomly select 2 or 3 body parts to swap
        k = 2 if random.random() < 0.5 else 3
        spa_part_idx = sorted(random.sample(range(self.part_num), k=k))

        # Collect pixel columns that belong to the selected parts
        spa_joint_idx = sorted(
            j for idx in spa_part_idx for j in range(idx * 16, (idx + 1) * 16)
        )

        N = motion.shape[0]
        idx = torch.arange(N)
        n = 1 if N <= 2 else torch.randint(1, N - 1, (1,)).item()
        randidx = (idx + n) % N

        xst = motion.clone()
        for i in range(N):
            if m_length[i] == 0 or m_length[randidx[i]] == 0:
                randidx[i] = i  # skip invalid samples
                continue

            length_pattern = m_length[randidx[i]]
            length_ori = m_length[i]

            # Temporally resize the source sample to match target length
            resize_pattern = torch.nn.functional.interpolate(
                motion[randidx[i], :, :length_pattern, :][None].transpose(2, 3),
                (motion.shape[-1], length_ori),
                mode='bilinear',
                align_corners=False,
            ).transpose(2, 3)

            xst[i, :, :length_ori, spa_joint_idx] = (
                resize_pattern[0, :, :, spa_joint_idx].clone()
            )

        part_text_embed[:, torch.tensor(spa_part_idx).cuda(), :] = (
            part_text_embed[randidx][:, torch.tensor(spa_part_idx).cuda(), :]
        )
        part_mask[:, spa_part_idx] = part_mask[randidx][:, spa_part_idx]

        lamb = len(spa_part_idx) / self.part_num
        return xst, part_text_embed, part_mask, lamb, randidx

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    
    def forward_train(
        self,
        motion,
        text,
        part_text=None,
        part_mask=None,
        stage_text=None,
        stage_mask=None,
        return_loss: bool = False,
        m_length=None,
    ):
        """
        Full training forward pass with all auxiliary losses.

        Returns (when return_loss=True and part_contrast enabled):
            global_clip_loss, mix_clip_loss, part_contrast_loss,
            mix_part_contrast_loss, stage_contrast_loss, local_loss, loss_order
        """

        def gumbel_max_trick(logits, tau=1.0):
            """Straight-through Gumbel-softmax argmax."""
            gumbels = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            gumbels = (logits / tau) + gumbels
            y_soft = gumbels.softmax(dim=-1)
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, index, 1.0)
            return index, y_hard

        motion_embeds, seq_motion_embeds = self.encode_motion(
            motion, m_length, return_seq=True
        )
        text_embeds = self.encode_text(text)

        if self.part:
            bs = motion_embeds[0].shape[0]
            part_text_embeds = self.encode_text(part_text).reshape(bs, self.part_num, -1)
            stage_text_embeds = self.encode_text(stage_text).reshape(bs, 4, -1)  # 4 = max event count

            mix_motion, mix_part_embeds, mix_part_mask, lamb, randidx = self.mix_part(
                motion, part_text_embeds, part_mask, m_length
            )
            mix_motion_embeds = self.encode_motion(mix_motion, m_length)

            # L2-normalise all embeddings
            for i in range(1 + self.part_num):
                motion_embeds[i] = F.normalize(motion_embeds[i], dim=-1)
                mix_motion_embeds[i] = F.normalize(mix_motion_embeds[i], dim=-1)
            part_text_embeds = F.normalize(part_text_embeds, dim=-1)
            stage_text_embeds = F.normalize(stage_text_embeds, dim=-1)
            mix_part_embeds = F.normalize(mix_part_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            seq_motion_embeds = F.normalize(seq_motion_embeds, dim=-1)

            # Interpolate mixed text embeddings for the swapped samples
            mix_text_embeds = text_embeds[randidx] * lamb + text_embeds * (1 - lamb)
            mix_text_embeds = F.normalize(mix_text_embeds, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds[0].t()) * logit_scale

            part_contrast_loss = 0.0
            mix_part_contrast_loss = 0.0
            stage_contrast_loss = 0.0

            # Part-level contrastive loss on original samples
            for i in range(self.part_num):
                pmask = (
                    part_mask[:, i].bool()
                    if part_mask is not None
                    else torch.zeros(bs, dtype=torch.bool).cuda()
                )
                if pmask.long().sum() >= bs - 1:
                    continue
                part_logits = (
                    torch.matmul(
                        part_text_embeds[~pmask, i, :],
                        motion_embeds[i + 1][~pmask, :].t(),
                    ) * logit_scale
                )
                part_contrast_loss += self.clip_loss(part_logits)

            # Part-level contrastive loss on mixed samples
            for i in range(self.part_num):
                pmask = (
                    mix_part_mask[:, i].bool()
                    if mix_part_mask is not None
                    else torch.zeros(bs, dtype=torch.bool).cuda()
                )
                if pmask.long().sum() >= bs - 1:
                    continue
                mix_part_logits = (
                    torch.matmul(
                        mix_part_embeds[~pmask, i, :],
                        mix_motion_embeds[i + 1][~pmask, :].t(),
                    ) * logit_scale
                )
                mix_part_contrast_loss += self.clip_loss(mix_part_logits)

            # Temporal stage ordering loss
            frame_mask = generate_length_mask_2d(m_length).float()
            mu_list = []
            # Gumbel-sampled motion clip feature for each stage, shape (B, D).
            # Used later for event-level local directional loss.
            stage_motion_clips = []
            for i in range(4):
                smask = ~stage_mask[:, i].bool()

                # Token-wise similarity between stage text and frame sequence
                logits = torch.einsum(
                    'ac,bvc->abv',
                    [stage_text_embeds[smask, i], seq_motion_embeds[smask]],
                )  # B_valid x B_valid x L

                sim_seq = logits.clone()
                logits = logits.masked_fill(
                    ~frame_mask[smask][None].repeat(logits.shape[0], 1, 1).bool(),
                    float('-inf'),
                )

                # Gumbel argmax: diagonal uses tight tau, off-diagonal uses soft tau
                idx_diag, _ = gumbel_max_trick(logits, tau=1.0)
                idx_other, _ = gumbel_max_trick(logits, tau=10.0)

                # Extract the frame selected for each valid sample (diagonal of idx_diag).
                # idx_diag: (B_valid, B_valid, 1) → diag gives (B_valid,) frame indices.
                B_valid = smask.sum()
                diag_frame_idx = idx_diag[
                    torch.arange(B_valid), torch.arange(B_valid), 0
                ]  # (B_valid,)
                # Gather the corresponding frame feature from seq_motion_embeds.
                clip_feat = seq_motion_embeds[smask][
                    torch.arange(B_valid), diag_frame_idx
                ]  # (B_valid, D)
                # Write into a full-batch tensor; invalid samples remain zero-filled.
                stage_clip_full = torch.zeros(bs, seq_motion_embeds.shape[-1], device=motion.device)
                stage_clip_full[smask] = clip_feat
                stage_motion_clips.append(stage_clip_full)
                logits_diag = torch.gather(logits, dim=-1, index=idx_diag)[..., 0]
                logits_other = torch.gather(logits, dim=-1, index=idx_other)[..., 0]

                diag_mask = torch.eye(logits_diag.shape[0]).cuda()
                stage_logits = logits_diag * diag_mask + logits_other * (1.0 - diag_mask)

                if len(stage_logits) > 1:
                    stage_contrast_loss += self.clip_loss(stage_logits * logit_scale)

                # Soft expected frame position for this stage
                sim_seq = torch.einsum(
                    'ac,avc->av',
                    [stage_text_embeds[smask, i], seq_motion_embeds[smask]],
                )  # B x L
                sim_seq = sim_seq.masked_fill(frame_mask[smask] == 0, float('-inf'))
                sim_seq = F.softmax(sim_seq / 3.0, dim=-1)  # B x L

                frame_idx = torch.arange(224 // 16).cuda()[None].repeat(sim_seq.shape[0], 1)
                mu = m_length.clone().float()
                mu[smask] = (frame_idx * sim_seq).sum(-1)
                mu_list.append(mu)

            # Penalise violations of temporal order across the 4 stages
            delta = 0.1 * (m_length // 16)
            loss_order = (
                F.relu(mu_list[0] - mu_list[1] + delta)
                + F.relu(mu_list[1] - mu_list[2] + delta)
                + F.relu(mu_list[2] - mu_list[3])
            ).mean()

            part_local_loss = local_direc_loss(
                motion_embeds[0],
                motion_embeds[1:],
                text_embeds,
                part_text_embeds.unbind(1),
                part_mask,
            )
            event_local_loss = local_direc_loss(
                motion_embeds[0],
                stage_motion_clips,                  # list of 4, each (B, D)
                text_embeds,
                list(stage_text_embeds.unbind(1)),   # list of 4, each (B, D)
                stage_mask,                          # (B, 4)
            )
            local_loss = part_local_loss + event_local_loss

            if return_loss:
                return (
                    self.clip_loss(logits_per_text),
                    self.clip_loss(
                        torch.matmul(mix_text_embeds, mix_motion_embeds[0].t()) * logit_scale
                    ),
                    part_contrast_loss / self.part_num,
                    mix_part_contrast_loss / self.part_num,
                    stage_contrast_loss / 4.0,
                    local_loss,
                    loss_order,
                )
            else:
                return (
                    motion_embeds,
                    text_embeds,
                    part_text_embeds,
                    self.clip_loss(logits_per_text),
                    part_contrast_loss / self.part_num,
                )

        else:
            # Baseline: global contrastive loss only
            motion_embeds = F.normalize(motion_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale

            if return_loss:
                return self.clip_loss(logits_per_text)
            else:
                return motion_embeds, text_embeds

    def forward(self, motion, text, part_text=None, part_mask=None, return_loss=False, length=None):
        """
        Inference forward pass (no mixed samples, no stage ordering).

        Args:
            motion:      Motion input tensor.
            text:        Tokenised text dict.
            part_text:   Tokenised part-level text annotations (part mode only).
            part_mask:   Missing-part boolean mask (part mode only).
            return_loss: Whether to return loss values.
            length:      Valid frame lengths.
        """
        motion_embeds = self.encode_motion(motion, length=length)
        text_embeds = self.encode_text(text)

        if self.part:
            bs = motion_embeds[0].shape[0]
            part_text_embeds = self.encode_text(part_text).reshape(bs, self.part_num, -1)

            for i in range(1 + self.part_num):
                motion_embeds[i] = F.normalize(motion_embeds[i], dim=-1)
            part_text_embeds = F.normalize(part_text_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds[0].t()) * logit_scale

            part_contrast_loss = 0.0
            for i in range(self.part_num):
                pmask = (
                    part_mask[:, i].bool()
                    if part_mask is not None
                    else torch.zeros(bs, dtype=torch.bool).cuda()
                )
                if pmask.long().sum() >= bs - 1:
                    return 0.0
                part_logits = (
                    torch.matmul(
                        part_text_embeds[~pmask, i, :],
                        motion_embeds[i + 1][~pmask, :].t(),
                    ) * logit_scale
                )
                part_contrast_loss += self.clip_loss(part_logits)

            if return_loss:
                return self.clip_loss(logits_per_text), part_contrast_loss / self.part_num
            else:
                return (
                    motion_embeds,
                    text_embeds,
                    part_text_embeds,
                    self.clip_loss(logits_per_text),
                    part_contrast_loss / self.part_num,
                )

        else:
            # Baseline: global contrastive loss only
            motion_embeds = F.normalize(motion_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale

            if return_loss:
                return self.clip_loss(logits_per_text)
            else:
                return motion_embeds, text_embeds