"""Evaluate a trained checkpoint on the CNN/DailyMail test set.

Metrics (on generated summaries, greedy decoding), each also computed for a Lead-3 baseline:
  1. ROUGE-1 / ROUGE-2 / ROUGE-L  - n-gram / LCS overlap, the standard CNN/DM benchmark metric.
  2. BERTScore F1 (rescaled)      - contextual-embedding similarity, credits paraphrases (abstractive output).
  3. METEOR                       - unigram alignment with stemming/synonyms, recall-weighted.
(Named evaluation.py, not evaluate.py, to avoid shadowing the HF `evaluate` package.)
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
import torch
import evaluate

from data import BOS, EOS, PAD, clean, load_split, loader, to_dataset, tokenizer
from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default="outputs", help="folder with model.pth + config.json from train.py")
p.add_argument("--data-dir", default="data")
p.add_argument("--n-test", type=int, default=2000, help="0 = full test set (11,490)")
p.add_argument("--gen-len", type=int, default=100)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--num-workers", type=int, default=4)
args = p.parse_args()

cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

model = build_model(len(tokenizer), PAD, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"]).to(device)
model.load_state_dict(torch.load(os.path.join(args.run_dir, "model.pth"), map_location=device))
model.eval()

test_df = load_split("test", args.data_dir)
if args.n_test:
    test_df = test_df.sample(args.n_test, random_state=cfg["seed"])
test_ds = to_dataset(test_df, cfg["max_src"], cfg["max_tgt"])
refs = list(test_df.highlights.map(clean))

preds = []
for b in loader(test_ds, args.batch_size, False, args.num_workers):
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        ys = model.generate(b["input_ids"].to(device), BOS, EOS, args.gen_len)
    preds += tokenizer.batch_decode(ys, skip_special_tokens=True)

# Lead-3 baseline (first 3 sentences of the article) gives the metrics a reference point
lead3 = [" ".join(re.split(r"(?<=[.!?])\s+", clean(a))[:3]) for a in test_df.article]

rouge, meteor, bertscore = evaluate.load("rouge"), evaluate.load("meteor"), evaluate.load("bertscore")


def score(hyps):
    r = rouge.compute(predictions=hyps, references=refs, use_stemmer=True)
    m = meteor.compute(predictions=hyps, references=refs)
    bs = bertscore.compute(predictions=hyps, references=refs, lang="en", batch_size=64,
                           rescale_with_baseline=True)  # raw scores crowd into 0.8-0.9
    return {"rouge1": r["rouge1"] * 100, "rouge2": r["rouge2"] * 100, "rougeL": r["rougeL"] * 100,
            "bertscore_f1": float(np.mean(bs["f1"])) * 100, "meteor": m["meteor"] * 100}


results = {"n_test": len(refs), "transformer": score(preds), "lead3_baseline": score(lead3)}
print(json.dumps(results, indent=2), flush=True)
json.dump(results, open(os.path.join(args.run_dir, "test_metrics.json"), "w"), indent=2)
pd.DataFrame({"article": test_df.article.map(clean), "reference": refs, "prediction": preds}) \
    .to_csv(os.path.join(args.run_dir, "test_predictions.csv"), index=False)
