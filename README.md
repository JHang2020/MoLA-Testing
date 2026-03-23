# Unified Human Motion Pretraining

> **Anonymous submission under review.** This repository contains the code release accompanying the paper submission. Model weights and full data preprocessing pipeline will be released upon acceptance.

---

## 🚀 Release Plan (TODO)

- [x] Full Model code (`mola.py`, `m2t_model.py`)
- [x] Training & evaluation scripts for motion-text retrieval
- [x] Training & evaluation scripts for downstream tasks
- [ ] LLM-augmented annotation data
- [ ] Pretrained model weights
- [ ] Demo & visualization resources


---

## 🛠️ Requirements

- Python 3.11
- PyTorch 2.0.1+
- CUDA 11.7+

```bash
pip install -r requirements.txt
```

### Dependencies
For motion to text evaluation
```
bash prepare/download_evaluators.sh
```
```
bash prepare/download_glove.sh
```


### ⚠️ Pretrained Language Models

The following models are loaded from HuggingFace at runtime. If you are in a restricted-network environment, download them manually and set the corresponding local paths in your config.

| Model | HuggingFace ID | Used for |
|---|---|---|
| DistilBERT | `distilbert/distilbert-base-uncased` | Text encoder |
| RoBERTa-Large | `FacebookAI/roberta-large` | BERT Score Evaluation |
| OPT-1.3B | `facebook/opt-1.3b` | Motion-to-text backbone |

To download manually:

```bash
# Example — repeat for each model
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-1.3b', local_dir='./pretrained_lm/opt-1.3b')"
```

Then update the model path to point to the local directory instead of the hub ID.

---

## 📂 Data Preparation

- Download **HumanML3D** and **KIT-ML** from the [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D).

- Download **BABEL-60** and **BABEL-120** dataset from [here](https://drive.google.com/drive/u/0/folders/1_SpOgtYCZBPAXoVz00Zhyk6tPRObUIiW). Then process them in HumanML3D style. 

- Organise the data directory as follows:

```
.
├── README.md
├── requirements.txt
├── conf/
├── scripts/
└── data/
    ├── HumanML3D/
    │   ├── new_joint_vecs/
    │   ├── new_joints/
    │   ├── part_texts/
    │   ├── event_texts/
    │   ├── texts/        
    │   └── ...
    └── KIT-ML/
        ├── new_joint_vecs/
        ├── new_joints/
        ├── part_texts/
        ├── event_texts/
        ├── texts/        
        └── ...
```

> **Note:** The LLM-augmented part-text and event-text annotations will be released as a separate download in the future. This is required for the full training objective.


- Compute dataset statistics (mean / variance) before training:
```bash
python scripts/cal_mean_var.py
```

---

## 💻 Training and Usage

### Pretraining
Stage 1 Training:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_mola_stage1.py dataset=HumanML3D exp_name=mola_stage1
```

Stage 2 Training:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_mola_stage2.py dataset=HumanML3D exp_name=mola_stage2 resume=checkpoints/mola_stage1/best_model.pt
```

### Motion-to-Text Training (> 40G GPU Memory)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_m2t.py dataset=HumanML3D exp_name=mola_stage2_m2t
```

---

## Evaluation

### Motion–Text Retrieval
HumanML3D Dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test.py dataset=HumanML3D exp_name=mola_stage2
```

KIT-ML Dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test.py dataset=KIT-ML exp_name=mola_stage2
```

### Motion-to-Text Captioning

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_m2t.py dataset=HumanML3D exp_name=mola_stage2_m2t resume=/path/to/your/saved/m2t_model
```

### Zero-Shot Action Recognition (BABEL)

BABEL-60 Dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/action_recognition.py dataset=BABEL-60 exp_name=mola_stage2
```

BABEL-120 Dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/action_recognition.py dataset=BABEL-120 exp_name=mola_stage2
```

---

## 📜 Citation

If you find this work useful, please consider citing the following work(s):

```bibtex
@inproceedings{zhangsgar2024,
  title     = {SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval},
  author    = {Zhang, Jiahang and Lin, Lilang and Yang, Shuai and Liu, Jiaying},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## 🙏 Acknowledgements

We sincerely thank the authors of previous works for releasing their codebases. Parts of our code are built upon [MotionPatches](https://github.com/line/MotionPatches), [TMR++](https://github.com/leorebensabath/TMRPlusPlus), [LaMP](https://github.com/gentlefress/LaMP/).

---

## 📄 Licence

This project is licensed under the MIT License. The pretrained language models (OPT, RoBERTa, DistilBERT) are subject to their respective licences on HuggingFace.