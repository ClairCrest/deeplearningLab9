#!/bin/bash
#SBATCH --job-name=lab_9
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpul40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# End-to-end: setup -> download -> train -> evaluate.
# Submit from ~/deep_learning:  sbatch slurm/lab_9.sbatch
# Extra args go to train.py, e.g.: sbatch slurm/lab_9.sbatch --epochs 5

set -euo pipefail

ROOT=$HOME/deep_learning
SRC=$ROOT/src/lab_9
DATA=$ROOT/data/lab_9
RUN_DIR=$ROOT/results/lab_9/$SLURM_JOB_ID

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $ROOT/env_dl/bin/activate

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== GPU ==="
nvidia-smi || echo "WARNING: nvidia-smi failed on $(hostname)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available on this node'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "=== [1/4] Install dependencies ==="   # no-op when already installed
if command -v uv >/dev/null; then
    uv pip install -q --python "$(which python)" -r $SRC/requirements.txt
else
    python -m pip install -q -r $SRC/requirements.txt
fi

echo "=== [2/4] Download dataset, tokenizer, metric models ==="   # skips anything already cached
python $SRC/prefetch.py "$DATA"

echo "=== [3/4] Train ==="
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python $SRC/train.py --data-dir "$DATA" --out-dir "$RUN_DIR" --num-workers "$SLURM_CPUS_PER_TASK" "$@"

echo "=== [4/4] Evaluate ==="
python $SRC/evaluation.py --data-dir "$DATA" --run-dir "$RUN_DIR" --num-workers "$SLURM_CPUS_PER_TASK"

echo "=== Results ==="
cat "$RUN_DIR/test_metrics.json"
echo "Outputs in: $RUN_DIR"
echo "Job finished on: $(date)"
