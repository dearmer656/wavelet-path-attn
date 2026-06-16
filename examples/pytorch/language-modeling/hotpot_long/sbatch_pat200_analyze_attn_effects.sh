#!/bin/bash
#SBATCH --job-name=pat200_analyze_attn_effects
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat200_analyze_attn_effects.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

OUTSUB="${OUTSUB:-full}"
RSTAR="${RSTAR:-6}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-200" --gpu "CPU" \
        --summary "PAT-200 Phase B attention-effect analysis + Gate 3 (R*=${RSTAR}, ${OUTSUB})" 2>/dev/null || true
}
trap '_slack $?' EXIT
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONUNBUFFERED=1
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long
DIR=analysis_outputs/qwab_scale_query_nmf/${OUTSUB}

python analyze_attention_effects.py \
    --effects $DIR/attention_effects_query_table.parquet \
    --out_dir $DIR --R ${RSTAR} --n_boot 1000 --seed 0

echo "=== Done: PAT-200 Phase B analysis (${OUTSUB}) ==="
