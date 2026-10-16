# CKU

Implementation of [Safety Alignment via Constrained Knowledge Unlearning](https://aclanthology.org/2025.acl-long.1240/) (ACL 2025).

## Installation

Use a Python 3.10–3.13 environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the following commands from the repository root.

## Download Models

1. Get [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and place it in `./models/Llama2-7b-chat-hf/`.
2. For Llama 3 experiments, get [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and place it in `./models/Meta-Llama-3-8B-Instruct/`.

## Prepare Data

Prepare the following files:

- `./data/calibration.json`: useful-knowledge examples for importance scoring.
- `./data/forget.json`: prompt-response pairs for unlearning.

Each file is a JSON list with `input` and `output` fields:

```json
[
  {"input": "Name a color.", "output": "Blue."}
]
```

The training datasets are not included in this repository.

## Obtain Masks

Compute importance scores on the calibration data:

```bash
python main.py score \
  --model ./models/Llama2-7b-chat-hf/ \
  --data ./data/calibration.json \
  --output ./outputs/scores.pt \
  --layer-start 8 --layer-end 12 \
  --num-samples 128
```

Generate the utility mask with a locking rate of 0.8:

```bash
python main.py mask \
  --scores ./outputs/scores.pt \
  --output ./outputs/masks.pt \
  --lock-ratio 0.8
```

The mask protects the highest-scored 80% of weight elements in each selected MLP projection. The layer range `[8, 12)` uses zero-based indices and excludes layer 12.

## CKU Training

After preparing the data and mask, run:

```bash
python main.py train \
  --model ./models/Llama2-7b-chat-hf/ \
  --data ./data/forget.json \
  --masks ./outputs/masks.pt \
  --output ./outputs/cku_model/ \
  --harmful-threshold 1.5 \
  --learning-rate 4e-6 \
  --batch-size 4 \
  --epochs 1
```

The trained model and tokenizer are saved to `./outputs/cku_model/`.

For Llama 3, replace the model path in both scoring and training commands with `./models/Meta-Llama-3-8B-Instruct/` and generate its mask separately.

## Cite

```bibtex
@inproceedings{shi-etal-2025-safety,
  title = {Safety Alignment via Constrained Knowledge Unlearning},
  author = {Shi, Zesheng and Zhou, Yucheng and Li, Jing and Jin, Yuxin and Li, Yu and He, Daojing and Liu, Fangming and Alharbi, Saleh and Yu, Jun and Zhang, Min},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2025},
  pages = {25515--25529},
  doi = {10.18653/v1/2025.acl-long.1240}
}
```
