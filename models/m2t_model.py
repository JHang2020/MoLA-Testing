import random

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, OPTForCausalLM

class MotionCaptionModel(nn.Module):
    """
    Motion-conditioned captioning model.

    A frozen MotionEncoder encodes skeleton sequences into a small set of
    prefix tokens that are projected into the LLM's embedding space and
    prepended to the text tokens before the forward pass.

    Architecture
    ------------
    MotionEncoder  →  ProjectionMLP  →  OPT (+ optional LoRA)

    Args:
        motion_encoder:         Pre-trained MotionEncoder (weights frozen).
        opt_model_name:         Path or HuggingFace hub name of the OPT checkpoint.
        tokenizer_path:         Path or hub name for the corresponding tokenizer.
        num_reg_tokens:         Number of register tokens produced by the encoder
                                (determines prefix length = 1 + num_reg_tokens).
        projection_hidden_size: Hidden dim of the two-layer projection MLP.
        lora_r:                 LoRA rank.
        freeze_llm:             Freeze LLM weights when use_lora=False.
        use_lora:               Apply LoRA adapters to the LLM.
    """

    # Prompt templates sampled randomly during training for robustness.
    PROMPT_TEMPLATES = [
        "Describe this motion:",
        "Action:",
        "Motion:",
        "What is happening?",
        "The person is",
        "This shows a person",
        "Someone is",
        "You can see a person",
        "A person is performing:",
        "The movement is:",
        "This action involves:",
        "Observed behavior:",
    ]

    def __init__(
        self,
        motion_encoder: nn.Module,
        opt_model_name: str = '/mnt/netdisk/zhangjh/Code/MotionPatches/opt-1.3b',#"facebook/opt-1.3b",
        tokenizer_path: str = '/mnt/netdisk/zhangjh/Code/MotionPatches/opt-1.3b',#"facebook/opt-1.3b",
        num_reg_tokens: int = 5,
        projection_hidden_size: int = 2048,
        lora_r: int = 16,
        freeze_llm: bool = True,
        use_lora: bool = True,
    ):
        super().__init__()

        # Motion encoder (kept frozen throughout training)
        self.motion_encoder = motion_encoder
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        # ClipModel projection outputs 256-dim embeddings
        motion_embed_dim = 256
        self.num_prefix_tokens = 1 + num_reg_tokens  # [CLS] + register tokens

        # Projection MLP: motion embedding space → LLM input embedding space
        opt_model = transformers.OPTForCausalLM.from_pretrained(opt_model_name)
        llm_embed_dim = opt_model.config.word_embed_proj_dim

        self.projection = nn.Sequential(
            nn.Linear(motion_embed_dim, projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, llm_embed_dim),
        )

        # Tokenizer
        self.opt_tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False
        )
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token
        self.opt_tokenizer.padding_side = "right"

        # Use newline as the EOS token (marks end of a caption line)
        self.eos_token = "\n"
        self.eos_token_id = self.opt_tokenizer(
            self.eos_token, add_special_tokens=False
        ).input_ids[0]
        self.opt_tokenizer.eos_token = self.eos_token
        self.opt_tokenizer.eos_token_id = self.eos_token_id

        # Reuse the already-loaded model rather than loading it a second time
        self.opt_model = opt_model
        self.opt_model.config.eos_token_id = self.eos_token_id
        self.opt_model.config.pad_token_id = self.opt_tokenizer.pad_token_id

        # LoRA / freeze
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                modules_to_save=[],
            )
            self.llm = get_peft_model(self.opt_model, lora_config)
            self.llm.print_trainable_parameters()
        else:
            self.llm = self.opt_model
            if freeze_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False

        self.max_caption_length = 64
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise projection MLP with small-variance normal weights."""
        for name, param in self.projection.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.005)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.ones_(param)

    def forward(self, motion_seq, captions=None, length=None):
        """
        Compute the captioning loss for a batch of motion–text pairs.

        Args:
            motion_seq: Motion patch tensor, shape (B, C, H, W).
            captions:   Ground-truth caption strings, List[str] of length B.
            length:     Valid frame lengths per sample, shape (B,).

        Returns:
            dict with key "loss" containing the scalar cross-entropy loss.
        """
        device = motion_seq.device
        embeddings = self.llm.get_input_embeddings()

        # Encode motion and project into LLM embedding space
        prefix_tokens_tuple = self.motion_encoder.encode_motion(motion_seq, length=length)
        prefix_tokens = torch.stack(prefix_tokens_tuple, dim=1)  # (B, num_prefix_tokens, D_motion)
        projected_tokens = self.projection(prefix_tokens)          # (B, num_prefix_tokens, D_llm)

        if captions is None:
            raise ValueError("Training requires captions.")

        # Randomly sample a prompt template for training robustness
        prompt_text = random.choice(self.PROMPT_TEMPLATES)
        prompted_captions = [prompt_text + cap.strip() + "\n" for cap in captions]

        B = len(motion_seq)
        text_inputs = self.opt_tokenizer(
            prompted_captions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_caption_length - 1,
        ).to(device)

        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        # Compute prompt length to mask prompt tokens from the loss
        prompt_token_len = self.opt_tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.shape[1]

        # Build full input: [motion prefix | text embeddings]
        text_embeds = embeddings(input_ids)                        # (B, T, D_llm)
        inputs_embeds = torch.cat([projected_tokens, text_embeds], dim=1)  # (B, P+T, D_llm)

        prefix_attention_mask = torch.ones(
            projected_tokens.shape[:2], dtype=torch.long, device=device
        )
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # Labels: ignore motion prefix and prompt tokens; supervise caption tokens only
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100              # ignore padding
        labels[:, : prompt_token_len + 1] = -100        # ignore prompt
        prefix_labels = torch.full(
            (B, projected_tokens.shape[1]), -100, device=device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        with autocast(dtype=torch.float16):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True,
                labels=labels,
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        motion_seq,
        length=None,
        prompt: str = "Describe this motion:",
        num_beams: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = None,
        repetition_penalty: float = 1.2,
        do_sample: bool = False,
    ):
        """
        Generate captions from motion sequences using beam search or sampling.

        Args:
            motion_seq:         Motion patch tensor, shape (B, C, H, W).
            length:             Valid frame lengths per sample, shape (B,).
            prompt:             Text prompt prepended before generation.
            num_beams:          Beam width (ignored when do_sample=True).
            max_new_tokens:     Maximum number of tokens to generate.
            temperature:        Sampling temperature (used only when do_sample=True).
            top_p:              Nucleus sampling threshold (used only when do_sample=True).
            repetition_penalty: Penalises repeated n-grams.
            do_sample:          If True, use sampling instead of beam search.

        Returns:
            List[str]: Decoded captions, one per sample in the batch.
        """
        device = motion_seq.device
        B = motion_seq.size(0)
        embeddings = self.llm.get_input_embeddings()

        # Encode motion
        prefix_tokens_tuple = self.motion_encoder.encode_motion(motion_seq, length=length)
        prefix_tokens = torch.stack(prefix_tokens_tuple, dim=1)  # (B, P, D_motion)
        projected_tokens = self.projection(prefix_tokens)          # (B, P, D_llm)

        # Build prompt embeddings
        if not prompt.endswith(" "):
            prompt = prompt + " "

        text_inputs = self.opt_tokenizer(
            [prompt] * B,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=20,
        ).to(device)

        text_embeds = embeddings(text_inputs.input_ids)
        inputs_embeds = torch.cat([projected_tokens, text_embeds], dim=1)

        prefix_len = projected_tokens.shape[1]
        combined_attention_mask = torch.cat([
            torch.ones(B, prefix_len, device=device, dtype=torch.long),
            text_inputs.attention_mask,
        ], dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1 if do_sample else num_beams,
            do_sample=do_sample,
            temperature=temperature,
            #top_p=top_p,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.opt_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )

        captions = self.opt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [cap.strip() for cap in captions]