"""Train the Transformer on CNN/DailyMail. Saves best checkpoint (by valid perplexity) + config.json + history.json."""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BOS, EOS, PAD, load_split, loader, to_dataset, tokenizer
from model import NoamOptim, build_model

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--out-dir", default="outputs")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--d-model", type=int, default=512)
p.add_argument("--n-layers", type=int, default=6)
p.add_argument("--n-heads", type=int, default=8)
p.add_argument("--d-ff", type=int, default=2048)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--max-src", type=int, default=512)
p.add_argument("--max-tgt", type=int, default=128)
p.add_argument("--warmup", type=int, default=4000)
p.add_argument("--lr-factor", type=float, default=1.0)
p.add_argument("--label-smoothing", type=float, default=0.1)
p.add_argument("--n-train", type=int, default=0, help="0 = all of train shard 0 (95,705)")
p.add_argument("--num-workers", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
os.makedirs(args.out_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)  # evaluate.py rebuilds the model from this
print(vars(args), "device:", device, "amp:", amp_dtype, flush=True)

train_df = load_split("train", args.data_dir)
if args.n_train:
    train_df = train_df.sample(args.n_train, random_state=args.seed)
train_ds = to_dataset(train_df, args.max_src, args.max_tgt)
valid_ds = to_dataset(load_split("validation", args.data_dir), args.max_src, args.max_tgt)
print(f"train {len(train_ds)}  valid {len(valid_ds)}", flush=True)
train_it = loader(train_ds, args.batch_size, True, args.num_workers)
valid_it = loader(valid_ds, args.batch_size, False, args.num_workers)

model = build_model(len(tokenizer), PAD, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.dropout).to(device)
print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
optimizer = NoamOptim(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                      args.d_model, args.lr_factor, args.warmup)
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)


def run_epoch(it, train, shuffle_src=False, max_batches=None):
    """Teacher-forced pass. Returns mean token NLL (no label smoothing) -> perplexity = exp(nll).
    shuffle_src pairs each summary with another article: if perplexity barely changes, the model ignores the source."""
    model.train(train)
    total, n_tok, t0 = 0.0, 0, time.time()
    with torch.set_grad_enabled(train):
        for i, b in enumerate(it):
            if i == max_batches:
                break
            src, tgt = b["input_ids"].to(device), b["labels"].to(device)
            if shuffle_src:
                src = src.roll(1, 0)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out, _ = model(src, tgt[:, :-1])
            out, gold = out.float().reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1)
            if train:
                optimizer.zero_grad()
                criterion(out, gold).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += F.nll_loss(out, gold, ignore_index=PAD, reduction="sum").item()
            n_tok += (gold != PAD).sum().item()
            if train and i % 200 == 0:
                print(f"  step {i}/{len(it)} loss {total / n_tok:.4f} lr {optimizer.get_lr():.2e} "
                      f"{time.time() - t0:.0f}s", flush=True)
    return total / n_tok


sample_src = torch.tensor([valid_ds[0]["input_ids"]], device=device)
best, history = float("inf"), []
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr, va = run_epoch(train_it, True), run_epoch(valid_it, False)
    vs = run_epoch(valid_it, False, shuffle_src=True, max_batches=50)
    history.append({"epoch": epoch, "train_ppl": math.exp(tr), "valid_ppl": math.exp(va), "valid_ppl_shuffled_src": math.exp(vs)})
    print(f"Epoch {epoch}: train ppl {math.exp(tr):.2f}  valid ppl {math.exp(va):.2f}  "
          f"valid ppl w/ wrong article {math.exp(vs):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    print("  sample:", tokenizer.decode(model.generate(sample_src, BOS, EOS, 100)[0], skip_special_tokens=True), flush=True)
    if va < best:
        best = va
        torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
        print("  saved best checkpoint", flush=True)
    json.dump(history, open(os.path.join(args.out_dir, "history.json"), "w"), indent=2)
