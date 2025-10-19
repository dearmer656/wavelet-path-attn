#! /bin/bash
#SBATCH --job-name=
#SBATCH --output=log_file/train/%j_s_0_10_softmix_wavelet_path_attn_from_s.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4



export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# 打印调试信息
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
echo 'Launching training/eval command'
echo '================= BEGIN RUN ================='


torchrun --nproc_per_node=2 --master_port=12422 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --share_freq_across_heads False \
--learning_rate 1e-4 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --block_size 512 --dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --num_train_epochs 30 --num_harmonics 1 \
--eval_strategy steps --eval_steps 250 --logging_dir ./s_0_10_softmix_wavelet_path_attn_from_s_log --logging_steps 250 --save_steps 5000 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/s_0_10_softmix_wavelet_path_attn_from_s \
--gradient_accumulation_steps 2 --b_unfreeze_step 5000 --pe_method vanilla --single_A_B False \
--use_beta_modulation False --use_wavelet_beta True --wavelet_mode softmix 

