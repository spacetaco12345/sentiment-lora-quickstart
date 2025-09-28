
# sentiment-lora-quickstart

Fast, copy‑paste repo to **train your own NLP model in ~15–25 minutes** using **LoRA fine‑tuning** on a small sentiment dataset (*rotten_tomatoes*). This is ideal when a job asks, “have you trained your own model?” — you can point to this repo and the resulting model artifact.

## What you get
- `train.py` — fine‑tunes **DistilBERT** with **LoRA** (parameter‑efficient).
- `infer.py` — run single‑text inference using the trained adapter.
- `requirements.txt` — minimal deps.
- Works on **Google Colab** or local GPU/CPU (GPU recommended).

---

## Quickstart (Colab or local)

```bash
# 1) Clone and install
pip install -r requirements.txt

# 2) Train (1 epoch on a small subset for speed)
python train.py --output_dir ./model --epochs 1 --train_samples 2000 --eval_samples 500 --batch_size 16

# 3) Inference
python infer.py --model_dir ./model --text "this movie was surprisingly fun and heartfelt"
```

Example output:
```
label: positive  (score ~0.9)
```

---

## Training details
- Base model: `distilbert-base-uncased`
- Dataset: `rotten_tomatoes` (binary sentiment)
- Method: **LoRA** adapters on attention projections (q, v) — lightweight & fast
- Default: 1 epoch on a 2k/500 train/eval subset (tweak via CLI flags)
- Metric: accuracy on the eval split

You can push the adapter to the Hugging Face Hub by setting `--push_to_hub` and logging in with a write token.

---

## Push to Hugging Face Hub (optional)

```bash
# Login once (stores your token)
python -c "from huggingface_hub import login; login()"

# Train and push
python train.py --output_dir ./model --epochs 1 --push_to_hub --repo_id spacetaco12345/rt-sentiment-lora
```

This publishes:
- the LoRA adapter weights/config,
- the tokenizer files,
- and a `README.md` card with metrics (from the run output).

---

## Folder contents
- `train.py` — loads dataset, tokenizes, applies LoRA, trains, evaluates, saves adapter & tokenizer.
- `infer.py` — loads base model + your LoRA adapter, runs a single‑text prediction.
- `requirements.txt` — tested with recent versions of Transformers/PEFT/Accelerate.

---

## Notes
- For “fastest possible” demo, keep defaults (1 epoch, small subset). For better accuracy, increase `--epochs` or remove the subset flags.
- LoRA makes only a tiny fraction of weights trainable, so it’s cheap and quick while still counting as “trained your own model.”
- If you want a super‑minimal classic ML baseline instead, swap to scikit‑learn + Iris in one file — but recruiters usually like to see modern DL tooling.
```

