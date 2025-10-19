#! /bin/bash
#SBATCH --job-name=
#SBATCH --output=log_file/test/%j_unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s_step80000_test.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:2
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


#################################################### for alibi #################################################### for alibi ###############################################                
torchrun --nproc_per_node=2 --master_port=12425 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "./runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s/checkpoint-80000" --share_freq_across_heads False \
--learning_rate 1e-4 --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 512 --dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s" \
--eval_strategy steps --logging_dir ./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s_log --logging_steps 250 --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s \
--gradient_accumulation_steps 2 --num_harmonics 1 --single_A_B False \
--use_beta_modulation False --use_wavelet_beta False --wavelet_mode db1 \
--wavelet_baseline_use True --init_theta 0.847


#################################################### for alibi #################################################### for alibi ###############################################                
torchrun --nproc_per_node=2 --master_port=12425 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "./runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s/checkpoint-80000" --share_freq_across_heads False \
--learning_rate 1e-4 --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 2048 --dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s" \
--eval_strategy steps --logging_dir ./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s_log --logging_steps 250 --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s \
--gradient_accumulation_steps 2 --num_harmonics 1 --single_A_B False \
--use_beta_modulation False --use_wavelet_beta False --wavelet_mode db1 \
--wavelet_baseline_use True --init_theta 0.847


#################################################### for alibi #################################################### for alibi ###############################################                
torchrun --nproc_per_node=2 --master_port=12425 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "./runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s/checkpoint-80000" --share_freq_across_heads False \
--learning_rate 1e-4 --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 4096 --dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s" \
--eval_strategy steps --logging_dir ./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s_log --logging_steps 250 --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s \
--gradient_accumulation_steps 2 --num_harmonics 1 --single_A_B False \
--use_beta_modulation False --use_wavelet_beta False --wavelet_mode db1 \
--wavelet_baseline_use True --init_theta 0.847


#################################################### for alibi #################################################### for alibi ###############################################                
torchrun --nproc_per_node=2 --master_port=12425 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "./runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s/checkpoint-80000" --share_freq_across_heads False \
--learning_rate 1e-4 --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s" \
--eval_strategy steps --logging_dir ./unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s_log --logging_steps 250 --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/unlearnable_wavelet_w_0_10_scale_01_shift_path_attn_from_s \
--gradient_accumulation_steps 2 --num_harmonics 1 --single_A_B False \
--use_beta_modulation False --use_wavelet_beta False --wavelet_mode db1 \
--wavelet_baseline_use True --init_theta 0.847


