
import argparse
import os
import numpy as np

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

def get_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune DistilBERT on Rotten Tomatoes")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default="./model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--train_samples", type=int, default=2000, help="subset size; use -1 for full")
    p.add_argument("--eval_samples", type=int, default=500, help="subset size; use -1 for full")
    p.add_argument("--fp16", action="store_true", help="enable fp16 if GPU supports")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--repo_id", type=str, default=None, help="HF repo name, e.g. user/rt-sentiment-lora")
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load dataset
    ds = load_dataset("rotten_tomatoes")
    if args.train_samples and args.train_samples > 0:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(args.train_samples))
    if args.eval_samples and args.eval_samples > 0:
        ds["test"] = ds["test"].shuffle(seed=42).select(range(args.eval_samples))

    # 2) Tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def _tok(b): return tok(b["text"], truncation=True)
    tokenized = ds.map(_tok, batched=True, remove_columns=["text"])

    base = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # 3) LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]  # DistilBERT attention
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # 4) Training args
    data_collator = DataCollatorWithPadding(tokenizer=tok)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean().item()}

    targs = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.repo_id if args.repo_id else None,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5) Train & evaluate
    trainer.train()
    eval_res = trainer.evaluate()
    print("Eval:", eval_res)

    # 6) Save adapter + tokenizer
    model.save_pretrained(args.output_dir)  # saves LoRA adapter
    tok.save_pretrained(args.output_dir)
    # also record base model name for inference
    with open(os.path.join(args.output_dir, "base_model.txt"), "w") as f:
        f.write(args.model_name)

    # 7) Optional push to hub
    if args.push_to_hub:
        trainer.push_to_hub()
        from huggingface_hub import HfApi
        if args.repo_id:
            print(f"Pushed to https://huggingface.co/{args.repo_id}")
        else:
            # Trainer sets a default repo id; we can't easily know it here without the login name.
            print("Pushed adapter to your Hugging Face Hub repo (see trainer logs for repo id).")

if __name__ == "__main__":
    main()
