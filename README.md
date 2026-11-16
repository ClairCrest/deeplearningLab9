# CPE 663 Lab 9: Transformer for Text Summarization (CNN/DailyMail)

Transformer encoder-decoder trained from scratch for abstractive summarization, evaluated with ROUGE, BERTScore and METEOR.

## Layout
```
src/lab_9/      data.py, model.py, train.py, evaluation.py, prefetch.py, requirements.txt
slurm/          lab_9.sbatch (install -> download -> train -> evaluate)
results/lab_9/  per-job config.json, history.json, test_metrics.json, test_predictions.csv
logs/           Slurm .out/.err for jobs 26980 and 27101
screenshots/    sacct / squeue output
report/         Lab9_Report_WarapolPratumta.pdf
```

## Run
From `~/deep_learning` on the cluster:
```bash
sbatch slurm/lab_9.sbatch            # extra args go to train.py, e.g. --epochs 5
```

## Results (test, 2,000 articles)
| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | METEOR |
|---|---|---|---|---|---|
| Job 26980 (separate embeddings) | 10.6 | 0.5 | 9.2 | -9.7 | 6.3 |
| Job 27101 (shared embeddings) | 29.04 | 9.09 | 20.54 | 11.97 | 20.34 |
| Lead-3 baseline | 39.60 | 17.19 | 24.67 | 25.12 | 38.07 |

Model checkpoints (`model.pth`, 250-375 MB) are not included because they exceed GitHub's file size limit.
