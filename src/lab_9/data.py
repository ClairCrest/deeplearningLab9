"""Data loading and preprocessing for CNN/DailyMail summarization."""
import os, re
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

URL = "https://huggingface.co/api/datasets/abisee/cnn_dailymail/parquet/1.0.0/{}/0.parquet"
TOKENIZER = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
PAD, BOS, EOS = tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id  # BERT: [CLS]=start, [SEP]=end


def load_split(split, data_dir="data"):
    """Download the parquet once, then read it from disk (compute nodes may be offline)."""
    path = os.path.join(data_dir, f"{split}.parquet")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        pd.read_parquet(URL.format(split)).to_parquet(path)
    return pd.read_parquet(path)


def clean(text):
    # strip the "(CNN) --" / "LONDON, England (CNN) --" dateline and collapse whitespace
    text = re.sub(r"^.{0,80}?\(CNN\)\s*(--)?\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def to_dataset(df, max_src, max_tgt):
    ds = Dataset.from_pandas(pd.DataFrame({"src": df.article.map(clean), "tgt": df.highlights.map(clean)}))

    def tok(b):
        out = tokenizer(b["src"], truncation=True, max_length=max_src)
        out["labels"] = tokenizer(b["tgt"], truncation=True, max_length=max_tgt)["input_ids"]
        return out
    return ds.map(tok, batched=True, remove_columns=["src", "tgt"])


_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=PAD, return_tensors="pt")


def loader(ds, batch_size, shuffle, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collator,
                      num_workers=num_workers, pin_memory=True)
