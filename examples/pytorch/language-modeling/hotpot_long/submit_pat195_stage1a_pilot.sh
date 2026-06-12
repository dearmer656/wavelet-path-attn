#!/bin/bash

set -euxo pipefail

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
TMP_ROOT="${BASE}/hotpot_long/tmp_entmax_eval/pat195_stage1a"
HOTPOT_SCRIPT="${BASE}/hotpot_long/eval_pat195_hotpot_uniform_a6000.sh"
XSUM_SCRIPT="${BASE}/hotpot_long/eval_pat195_xsum_rouge_a6000.sh"

mkdir -p "${TMP_ROOT}"

alphas=(1.05 1.1 1.2 1.5)
submitted=()

find_hotpot_baseline_dir() {
    local ckpt="$1"
    local readme
    while IFS= read -r readme; do
        if grep -Fqx "base_model: ${ckpt}" "${readme}"; then
            local out_dir
            out_dir="$(dirname "${readme}")"
            if [ -f "${out_dir}/eval_results.json" ]; then
                printf '%s\n' "${out_dir}"
                return 0
            fi
        fi
    done < <(find "${BASE}/hotpot_long/results_uniform" -path '*/L4096/README.md' | sort)
    return 1
}

xsum_baseline_exists() {
    local run_dir="$1"
    compgen -G "${run_dir}/ckpt_eval_xsum_rouge/xsum_L1536*/eval_results.json" > /dev/null
}

write_entmax_cfg() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local alpha="$3"
    mkdir -p "$(dirname "${dst_cfg}")"
    cp "${src_cfg}" "${dst_cfg}"
    # src_cfg may not end with a newline (e.g. PA_only's supply_model.cfg),
    # which would otherwise concatenate onto its last line and corrupt parsing.
    printf '\n' >> "${dst_cfg}"
    {
        printf 'attn_norm=entmax\n'
        printf 'entmax_alpha=%s\n' "${alpha}"
        printf 'entmax_scope=all\n'
    } >> "${dst_cfg}"
}

write_softmax_cfg_copy() {
    local src_cfg="$1"
    local dst_cfg="$2"
    mkdir -p "$(dirname "${dst_cfg}")"
    cp "${src_cfg}" "${dst_cfg}"
}

submit_job() {
    local job_id
    job_id="$(sbatch --parsable "$@")"
    submitted+=("${job_id}")
}

models=(
    "pa_only_s42|${BASE}/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000|${BASE}/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42|triton"
    "qwab_s42|${BASE}/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000|${BASE}/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach|pytorch"
)

for model_spec in "${models[@]}"; do
    IFS='|' read -r model_key ckpt run_dir path_attn_impl <<< "${model_spec}"
    base_cfg="${run_dir}/supply_model.cfg"

    if [ ! -f "${base_cfg}" ]; then
        echo "missing base cfg: ${base_cfg}" >&2
        exit 1
    fi

    hotpot_baseline_dir="$(find_hotpot_baseline_dir "${ckpt}" || true)"
    if [ -z "${hotpot_baseline_dir}" ]; then
        hotpot_softmax_cfg="${TMP_ROOT}/${model_key}_hotpot_softmax/supply_model.cfg"
        write_softmax_cfg_copy "${base_cfg}" "${hotpot_softmax_cfg}"
        hotpot_softmax_out="${BASE}/hotpot_long/results_uniform/pat195_stage1a_${model_key}_softmax/L4096"
        submit_job "${HOTPOT_SCRIPT}" "${ckpt}" "${hotpot_softmax_cfg}" "${hotpot_softmax_out}" "${path_attn_impl}"
    fi

    if ! xsum_baseline_exists "${run_dir}"; then
        xsum_softmax_cfg="${TMP_ROOT}/${model_key}_xsum_softmax/supply_model.cfg"
        write_softmax_cfg_copy "${base_cfg}" "${xsum_softmax_cfg}"
        xsum_softmax_out="${run_dir}/ckpt_eval_xsum_rouge/pat195_stage1a_${model_key}_softmax_L1536"
        submit_job "${XSUM_SCRIPT}" "${ckpt}" "${xsum_softmax_cfg}" "${xsum_softmax_out}" "${path_attn_impl}"
    fi

    for alpha in "${alphas[@]}"; do
        alpha_tag="${alpha//./p}"

        hotpot_cfg="${TMP_ROOT}/${model_key}_hotpot_entmax_a${alpha_tag}/supply_model.cfg"
        write_entmax_cfg "${base_cfg}" "${hotpot_cfg}" "${alpha}"
        hotpot_out="${BASE}/hotpot_long/results_uniform/pat195_stage1a_${model_key}_entmax_a${alpha_tag}/L4096"
        submit_job "${HOTPOT_SCRIPT}" "${ckpt}" "${hotpot_cfg}" "${hotpot_out}" "${path_attn_impl}"

        xsum_cfg="${TMP_ROOT}/${model_key}_xsum_entmax_a${alpha_tag}/supply_model.cfg"
        write_entmax_cfg "${base_cfg}" "${xsum_cfg}" "${alpha}"
        xsum_out="${run_dir}/ckpt_eval_xsum_rouge/pat195_stage1a_${model_key}_entmax_a${alpha_tag}_L1536"
        submit_job "${XSUM_SCRIPT}" "${ckpt}" "${xsum_cfg}" "${xsum_out}" "${path_attn_impl}"
    done
done

printf '%s\n' "${submitted[@]}"
