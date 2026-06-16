#!/bin/bash
#SBATCH --job-name=pat200_analyze_nmf
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat200_analyze_nmf.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

# --- args: OUTSUB (subdir under qwab_scale_query_nmf) ---
OUTSUB="${OUTSUB:-full}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-200" \
        --gpu "CPU" \
        --summary "PAT-200 Phase A NMF analysis (R=2..8 sweep + enrichment + controls + gates) on ${OUTSUB}" 2>/dev/null || true
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONUNBUFFERED=1

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long
DIR=analysis_outputs/qwab_scale_query_nmf/${OUTSUB}

# --- table sanity pre-check: pi_null + sum(pi_s) ~ 1, no NaN in X ---
python - "$DIR/router_scale_query_table.parquet" <<'PY'
import sys, pandas as pd, numpy as np
df = pd.read_parquet(sys.argv[1])
K = sum(1 for c in df.columns if c.startswith("pi_s") and not c.startswith("pi_s_"))
pis = [f"pi_s{j+1}" for j in range(K)]
tot = df["pi_null"].values + df[pis].sum(axis=1).values
print(f"rows={len(df)} K={K} pi-sum mean={tot.mean():.4f} min={tot.min():.4f} max={tot.max():.4f}")
assert np.allclose(tot, 1.0, atol=1e-3), "pi_null + sum(pi_s) != 1"
xcols = [f"norm_pi_s{j+1}" for j in range(K)]
assert not df[xcols].isna().any().any(), "NaN in norm_pi_s"
print("role counts:", df["role"].value_counts().to_dict())
print("PRECHECK OK")
PY

python analyze_scale_query_nmf.py \
    --table $DIR/router_scale_query_table.parquet \
    --out_dir $DIR \
    --n_seeds 5 --n_perm 200 --top_frac 0.05 --seed 0

echo "=== Done: PAT-200 NMF analysis (${OUTSUB}) ==="
