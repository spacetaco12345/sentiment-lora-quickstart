
import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def get_args():
    p = argparse.ArgumentParser(description="Run inference with a LoRA adapter")
    p.add_argument("--model_dir", type=str, default="./model", help="folder with adapter + tokenizer + base_model.txt")
    p.add_argument("--text", type=str, required=True)
    return p.parse_args()

def main():
    args = get_args()

    base_model_name = "distilbert-base-uncased"
    base_path = os.path.join(args.model_dir, "base_model.txt")
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            base_model_name = f.read().strip()

    tok = AutoTokenizer.from_pretrained(args.model_dir)  # loads tokenizer from save
    base = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base, args.model_dir)
    model.eval()

    inputs = tok(args.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        pred = int(torch.argmax(logits, dim=-1).item())

    label_map = {0: "negative", 1: "positive"}
    print(f"label: {label_map[pred]}  (score {probs[pred]:.3f})")

if __name__ == "__main__":
    main()
