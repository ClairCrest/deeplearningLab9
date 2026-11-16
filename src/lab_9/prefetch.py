"""Run once on a login node (has internet): caches dataset, tokenizer and metric models for offline compute nodes."""
import sys
import evaluate
from data import load_split  # importing data also downloads the tokenizer

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
for s in ("train", "validation", "test"):
    print(s, load_split(s, data_dir).shape)
evaluate.load("rouge"); evaluate.load("meteor")
evaluate.load("bertscore").compute(predictions=["a"], references=["a"], lang="en", rescale_with_baseline=True)
print("prefetch done; on offline compute nodes set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1")
