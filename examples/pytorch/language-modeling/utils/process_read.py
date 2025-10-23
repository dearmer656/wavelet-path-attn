import subprocess
import pdb
import argparse
from glob import glob
import time
import os
import shutil
import json
def read_process(args, id):
    with open(f"log_file/output_{id}.txt", "r") as file:
        lines = file.readlines()

    # 去除空行，获取有内容的最后一行
    last_non_empty_line = None
    for line in reversed(lines):
        if line.strip():  # 检查行是否为空
            last_non_empty_line = line.strip()
            break
    message = f"{args.project_name} ID: {id} {last_non_empty_line}"
    curl_command = f'curl -X POST --data-urlencode "payload={{\\"channel\\": \\"shy-program-notification\\", \\"text\\": \\"{message}\\"}}" https://hooks.slack.com/services/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'
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    
def generate_torchrun_command(args, id, account_text):

    files_list = glob(f'{args.project_name}/*')
    key_word = 'checkpoint'
    # 解析 project_name
    # pdb.set_trace()
    params = args.project_name.split('_')
    max_num = 0
    base_select = args.base_select
    for file in files_list:
        sep_file_name = file.split('/')[-1]
        if key_word in sep_file_name:
            num = int(sep_file_name.split('-')[-1])
            if num > max_num:
                max_num = num
    # 提取参数
    # try:
    #     random_length, block_size = params[0].split('-')
    #     if '___' in random_length or '***' in random_length or '+++' in random_length:
    #         random_length = random_length[3:]
    #     relative_type = params[1][4:]
    #     pe_method = params[3]
    #     max_digits = params[4]
    #     ln_switch = params[5]
    #     seq_pe_layer_num = params[6]
    #     attention_direction = params[7]
    #     solo_ln_switch = params[8]
    # except:
    block_size = 512
    pe_method = 'alibi'
    relative_type = 2
    max_digits = 5
    ln_switch = 'True'
    random_length = 10000
    attention_direction = 'single'
    seq_pe_layer_num = 2
    solo_ln_switch = 'True'
    dataset = args.dataset
    relative_type = args.relative_type
    # 检查是否包含 ctt_loss
    contrastive_loss_switch = True if "ctt_loss" in args.project_name else False
    random = "True" if ('randomized' in args.project_name.lower() or 'random' in args.project_name.lower()) else "False"

    pad = 10
    if 'pad0' in args.project_name or '0pad' in args.project_name:
        pad = 0
    if args.visualization:
        block_size = 8192
    if pe_method == 'rotary':
        random == False
    # 生成命令
    if contrastive_loss_switch:
        random = True
    if 'rotary' in args.project_name:
        random = False
    if 'mse' in args.project_name:
        transfer_loss_select = 'mse'
    elif 'kl' in args.project_name:
        transfer_loss_select = 'kl'
    else:
        transfer_loss_select = 'kl'
        # raise ValueError('please make the project name containing transfer_loss_select name')
    
    smoothing_switch = True if 'smoothing' in args.project_name else False
    model_type = 'gpt2'
    if 'roformer' in args.project_name.lower():
        model_type = 'roformer'
    elif 'vit_' in args.project_name.lower():
        model_type = 'vit'
    # if dataset == 'needle':
    #     dataset = 'needle_injected_dataset'
    # else:
    #     dataset = 'wikitext'
    if args.steps != 0:
        max_num = args.steps
    # if 'checkpoint' not in args.project_name:
    #     args.project_name = args.project_name + '_checkpoint'
    if 'subtractive' in args.project_name or 'subtract' in args.project_name:
        subtractive_switch = True
    else:
        subtractive_switch = False
    pe_method = args.pe_method if args.pe_method != 'none' else 'relative'          
    learn_scale = '--learn_scale True' if args.learn_scale else '' 
    if model_type == 'vit':
        pe_method = args.pe_method
        root_path = '/project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/image-classification/'
        if args.debug:
            command = f"""
srun -p gpu_intr --gres=gpu:{args.GPU}:1 --pty {root_path}run_image_classification.py --model_type vit --dataset_name {args.dataset} \\
--output_dir outputs --remove_unused_columns False --label_column_name label --resolution {args.resolution} \\
--do_train --do_eval --learning_rate {args.lr_rate}  --lr_scheduler_type {args.lr_scheduler_type} --max_steps 2500 --per_device_train_batch_size {args.batch_size} \\
--per_device_eval_batch_size {args.batch_size} --logging_strategy steps --logging_steps 250 --eval_strategy steps --eval_steps 250 --save_strategy steps --save_steps 250 --output_dir "{root_path}{args.project_name}" --logging_dir {root_path}{args.project_name}_log \\
--load_best_model_at_end True     --save_total_limit 3 --seed 42 --overwrite_output_dir \\
--pe_method {pe_method}
"""
            print('\n\ngenerated command is: \n', command, '\n\n')
            exit()         
        else:
            command = f"""
torchrun --nproc_per_node={args.per_node}  /project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/image-classification/run_image_classification.py --model_type vit --dataset_name {args.dataset} \\
--output_dir outputs --remove_unused_columns False --label_column_name label --resolution {args.resolution} \\
--do_train     --do_eval --learning_rate {args.lr_rate}  --lr_scheduler_type {args.lr_scheduler_type}  --max_steps 10000     --per_device_train_batch_size {args.batch_size} \\
--per_device_eval_batch_size {args.batch_size}     --logging_strategy steps     --logging_steps 1000     --eval_strategy steps --eval_steps 1000 --save_strategy steps --save_steps 1000 --output_dir "{root_path}{args.project_name}" --logging_dir "{root_path}{args.project_name}"_log \\
--load_best_model_at_end True     --save_total_limit 3 --seed 42 --overwrite_output_dir \\
--pe_method {pe_method} --dataloader_num_workers {args.dataloader_num_workers}
"""
        print('\n\ngenerated command is: \n' + command, '\n\n')
        return command
      
    if len(args.PE_pretrain_file) != 0:
        command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 2000 --random {random} --random_length {random_length} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} --requires_grad {args.requires_grad} \\
--PE_pretrain_file {args.PE_pretrain_file} --PE_training_start_step {args.PE_training_start_step} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --kl_beta {args.kl_beta} --report_to none
"""
    
    else:
        if args.test:
            # if 'needle' in dataset:
            #     max_num = 3000
            if args.get_grad:
                command = ""
                # pdb.set_trace()
                for file in files_list:
                    cur_command = f"""
    srun -p gpu_intr --gres=gpu:a6000:1 --pty ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --resume_from_checkpoint {file} \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
    --dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 2000 --random {random} --random_length {random_length} \\
    --visualization_switch {args.visualization} --visualization_file_name {args.project_name} \\
    --contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none --get_grad True
    """
                    command = command + cur_command + '\n'
                print(command)
                exit()
            else:
                if args.visualization:
                    print('checkpoint: checkpoint-', max_num)
                    command = f"""
    torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name needle_injected_dataset_{block_size} \\
    --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    --visualization_switch True --visualization_file_name {args.project_name} \\
    --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --base_select {args.base_select} --report_to none \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --report_to none \\
    --PE_embd_dim {args.PE_embd_dim}
    """
                else:
                    command = ''
                    if dataset in ['wikitext', 'xsum']:
                        random_length = 10000
                        if args.entire_test:
                            for file in files_list:
                                if 'checkpoint-' not in file or int(file.split('-')[-1]) < args.test_start_step or int(file.split('-')[-1]) > args.test_end_step:
                                    continue
                                cur_command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 512 --dataset_name wikitext \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch False --visualization_file_name {args.project_name} \\
--contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
--base_select {args.base_select} --report_to none \\
--PE_embd_dim {args.PE_embd_dim}

torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 1024 --dataset_name wikitext \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch False --visualization_file_name {args.project_name} \\
--contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
--base_select {args.base_select} --report_to none \\
--PE_embd_dim {args.PE_embd_dim}

torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 2048 --dataset_name wikitext \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch False --visualization_file_name {args.project_name} \\
--contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
--base_select {args.base_select} --report_to none \\
--PE_embd_dim {args.PE_embd_dim}

torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 4096 --dataset_name wikitext \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch False --visualization_file_name {args.project_name} \\
--contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
--base_select {args.base_select} --report_to none \\
--PE_embd_dim {args.PE_embd_dim}

torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name wikitext \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch False --visualization_file_name {args.project_name} \\
--contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
--base_select {args.base_select} --report_to none \\
--PE_embd_dim {args.PE_embd_dim}
"""
                                command = command + cur_command + '\n'                         
                        # for base_select in [4300, 10150, 16000, 20050, 27000, 55500, 84000]:
                        # for base_select in [100000, 120000, 140000, 160000, 180000, 200000]:
                        else:
                            learn_scale = '--learn_scale True' if args.learn_scale else ''
                            phase_shift='--phase_shift True' if args.phase_shift else ''
                            position_shift='--position_shift True' if args.position_shift else ''
                            decay_matrix = '--decay_matrix True' if args.decay_matrix else ''
                            scale_reverse = '--scale_reverse True' if args.scale_reverse else ''
                            decay_rate = f"--decay_rate {args.decay_rate}"                            
                            new_command = ""
                            
                            # length_list = [512] if args.get_qk_l2 else [128, 256, 512, 1024, 2048, 4096, 8192]
                            # length_list = [512] if args.get_qk_l2 else [512, 1024, 2048, 4096, 8196]
                            # length_list = [512] if args.get_qk_l2 else [1536, 2048, 4096]
                            # length_list = [512] if args.get_qk_l2 else [512, 1024, 2048, 4096]
                            # length_list = [512] if args.get_qk_l2 else [512, 8192, 10240]
                            # length_list = [512] if args.get_qk_l2 else [512]
                            length_list = [512] if args.get_qk_l2 else [512, 2048, 4096, 8192]
                            # length_list = [512] if args.get_qk_l2 else [512]
                            
                            for block_size in length_list:
                            # for block_size in [1024, 2048, 4096, 8192]:
                            # for block_size in [512]:
                            # for block_size in [128, 256, 512]:
                                new_command = f"""
#################################################### for alibi #################################################### for alibi ###############################################                
torchrun --nproc_per_node={args.per_node} --master_port=12425 ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./runs/{args.project_name}/checkpoint-{max_num}" --share_freq_across_heads {args.share_freq_across_heads} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--eval_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --attn_implementation {args.attn_implementation} --path_use_qk_norm false \\
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/{args.project_name} \\
--gradient_accumulation_steps {args.gradient_accumulation_steps} --num_harmonics {args.num_harmonics} --single_A_B {args.single_A_B} \\
--use_beta_modulation {args.use_beta_modulation} --use_soft_wavelet_fox {args.use_soft_wavelet_fox} --wavelet_mode {args.wavelet_mode} \\
--wavelet_baseline_use {args.wavelet_baseline_use} --init_theta {args.init_theta} --use_forget_gate {args.use_forget_gate}
"""
    #################################################### for alibi #################################################### for alibi ###############################################    

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 512 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 1024 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 2048 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 4096 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 12288 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 16384 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --hard_negative_switch False \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation} \\
    # --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 2  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name wikitext \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch True --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad}

                                command += new_command + '\n'
                    elif 'qa' in dataset:
                        command = ''
                        # for ro in [0.01, 0.05, 0.1, 0.4, 0.7, 0.05]:
                        # for ro in [0.05]:
                        #     # max_num_dict = {0.01:[2000], 0.05:[5000], 0.1:[2000, 10000], 0.4:[5000, 10000], 0.7:[5000, 10000], 1.0:[2000, 5000, 10000], 0.05:{250, 2000, 5000, 10000}}
                        #     max_num_dict = {0.05:{2000}}
                        #     for max_num in max_num_dict[ro]:
                        #         params_phrase = args.project_name.split('_')
                        #         for ph in range(len(params_phrase)):
                        #             if 'ctt' in params_phrase[ph]:
                        #                 continue
                        #             if 'ct' in params_phrase[ph]:
                        #                 params_phrase[ph] = f'{ro}ct'
                        #         project_name = '_'.join(params_phrase)

                        #     length_list = [1024, 2048]
                        #     test_batch_size = 16
                        # for step in range(2000, 40001, 2000):
                        phase_shift='--phase_shift True' if args.phase_shift else ''
                        position_shift='--position_shift True' if args.position_shift else ''
                        decay_matrix = '--decay_matrix True' if args.decay_matrix else ''
                        scale_reverse = '--scale_reverse True' if args.scale_reverse else ''
                        decay_rate = f"--decay_rate {args.decay_rate}"                        
                        dataset_name = 'qa_1' if 'qa_1' in dataset else 'qa_2'
                        if args.extra_params:
                            length_list = args.extra_params.strip().split()
                            length_list = [int(i) for i in length_list]
                        # length_list = [1536, 2048, 4096]
                        # for block_size in [512, 1024, 2048, 4096, 8192, 12288, 16384]:
                        # for block_size in length_list:
                        for block_size in [512, 1024, 1536, 2048, 4096]:
                        # for block_size in [8192]:
                            test_batch_size = 16 // (block_size // 512)
                            test_batch_size = max(2, test_batch_size)
                            cur_command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path ./runs/{args.project_name}/checkpoint-{max_num} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {test_batch_size} --block_size {block_size} --dataset_name {dataset_name}_{block_size} \\
--dataset_config_name wikitext-103-raw-v1 --do_eval --overwritef_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--eval_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --attn_implementation {args.attn_implementation} --path_use_qk_norm false \\
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/{args.project_name} \\
--gradient_accumulation_steps {args.gradient_accumulation_steps} --b_unfreeze_step {args.b_unfreeze_step} --num_harmonics {args.num_harmonics}
                            """
                            command = command + cur_command + '\n'
                    else:
                        # max_num = 1000
                        random_length = 10000
                        if args.entire_test:
                            command = ""
                            for file in files_list:
                                if 'checkpoint-' not in file or int(file.split('-')[-1]) < args.test_start_step or int(file.split('-')[-1]) > args.test_end_step:
                                    continue
                                cur_command = f"""
    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 512 --dataset_name fixed_needle_injected_dataset_512 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 1024 --dataset_name fixed_needle_injected_dataset_1024 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 2048 --dataset_name fixed_needle_injected_dataset_2048 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 4096 --dataset_name fixed_needle_injected_dataset_4096 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 8192 --dataset_name fixed_needle_injected_dataset_8192 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} --max_position_embeddings {args.max_position_embeddings} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 12288 --dataset_name fixed_needle_injected_dataset_12288 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name}  --max_position_embeddings {args.max_position_embeddings} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none

    torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path {file} \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size 4  --per_device_eval_batch_size {args.batch_size} --block_size 16384 --dataset_name fixed_needle_injected_dataset_16384 \\
    --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    --visualization_switch False --visualization_file_name {args.project_name} \\
    --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    --base_select {args.base_select} --report_to none

    """
                                
                                command = command + cur_command + '\n'
                        else:
                            command = f"""
    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 512 --dataset_name fixed_needle_injected_dataset_512 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 1024 --dataset_name fixed_needle_injected_dataset_1024 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 2048 --dataset_name fixed_needle_injected_dataset_2048 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 4096 --dataset_name fixed_needle_injected_dataset_4096 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 8192 --dataset_name fixed_needle_injected_dataset_8192 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    # torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    # --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 12288 --dataset_name fixed_needle_injected_dataset_12288 \\
    # --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    # --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    # --visualization_switch False --visualization_file_name {args.project_name} \\
    # --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings 16384 \\
    # --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    # --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    # --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path "./{args.project_name}/checkpoint-{max_num}" \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size 1  --per_device_eval_batch_size 1 --block_size 16384 --dataset_name fixed_needle_injected_dataset_16384 \\
    --dataset_config_name wikitext-103-raw-v1 --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random False --random_length {random_length} \\
    --visualization_switch False --visualization_file_name {args.project_name} --interpolation {args.interpolation} \\
    --contrastive_loss_switch False --temperature 1 --ctt_loss_ratio 0.1 --max_position_embeddings {args.max_position_embeddings} \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} \\
    --base_select {args.base_select} --report_to none --pivots_num 1 --KL_batch_size 1 --interpolation {args.interpolation}

    """

            ##!!!!
        else:
            if args.resume:
                if max_num != 0:
                    command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --transfer_loss_select {transfer_loss_select} --subtractive_ratio {args.subtractive_ratio} --detach {args.detach} --resume_from_checkpoint "./{args.project_name}/checkpoint-{max_num}" \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} --shift {args.shift} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" --KL_batch_size {args.KL_batch_size} --interpolation {args.interpolation} \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 5000 --random {random} --random_length {random_length} --contrastive_num_list_len {args.contrastive_num_list_len} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} --alpha {args.alpha} --l2_lambda {args.l2_lambda} --base_select {args.base_select} --report_to none --PE_training_start_step {args.PE_training_start_step} \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} --subtractive_switch {subtractive_switch} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} --max_position_embeddings {args.max_position_embeddings} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none --pivots_num {args.pivots_num} \\
--PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline} --transfer_main_ratio {args.transfer_main_ratio} {learn_scale} --extra_bias {args.extra_bias}
"""

# torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --resume_from_checkpoint "./{args.project_name}/checkpoint-{max_num}" \\
# --learning_rate {args.lr_rate} --per_device_train_batch_size 16  --per_device_eval_batch_size 16 --block_size {block_size} --dataset_name {dataset} \\
# --dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
# --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 2500 --random {random} --random_length {random_length} \\
# --visualization_switch {args.visualization} --visualization_file_name {args.project_name} \\
# --contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio 0.1 --interpolation {args.interpolation} \\
# --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
# --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad}

                else:
                    command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --transfer_loss_select {transfer_loss_select} --subtractive_ratio {args.subtractive_ratio} --detach {args.detach} --resume_from_checkpoint "./{args.project_name}/checkpoint-20000" \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} --shift {args.shift} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" --KL_batch_size {args.KL_batch_size} --interpolation {args.interpolation} \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps 5000 --random {random} --random_length {random_length} --contrastive_num_list_len {args.contrastive_num_list_len} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} --alpha {args.alpha} --l2_lambda {args.l2_lambda} --base_select {args.base_select} --report_to none --PE_training_start_step {args.PE_training_start_step} \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} --subtractive_switch {subtractive_switch} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} --max_position_embeddings {args.max_position_embeddings} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none --pivots_num {args.pivots_num} \\
--PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline} --transfer_main_ratio {args.transfer_main_ratio} {learn_scale} --extra_bias {args.extra_bias}
"""
            else:
                if 'pretrain' in args.project_name:
                    if 'MSE' in args.project_name:
                        command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --transfer_loss_select {transfer_loss_select} \\
--learning_rate 1e-3 --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_num_feature_pretrain True --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 200 --random {random} --random_length {random_length} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} --shift {args.shift}  --detach {args.detach}  \\
--warmup_steps {args.warmup_step} --alpha {args.alpha} --l2_lambda {args.l2_lambda} --interpolation {args.interpolation} --transfer_main_ratio {args.transfer_main_ratio} \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none \\
--PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline}
"""                     
                    else:
                    
                        command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --transfer_loss_select {transfer_loss_select} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_num_feature_pretrain True --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 100 --save_steps 2500 --random {random} --random_length {random_length} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} --shift {args.shift} --interpolation {args.interpolation} --detach {args.detach}  \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none \\
--PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline} --transfer_main_ratio {args.transfer_main_ratio}
"""                               
                
                else:
                    phase_shift='--phase_shift True' if args.phase_shift else ''
                    position_shift='--position_shift True' if args.position_shift else ''
                    decay_matrix = '--decay_matrix True' if args.decay_matrix else ''
                    scale_reverse = '--scale_reverse True' if args.scale_reverse else ''
                    decay_rate = f"--decay_rate {args.decay_rate}"
                    resume_from_checkpoint = f"--resume_from_checkpoint {args.resume_from_checkpoint}" if args.resume_from_checkpoint else ''
                    random = "True" if ('randomized' in args.project_name.lower() or 'random' in args.project_name.lower()) else "False"
                    pre_train_file = f"--model_name_or_path {args.pre_train_file}" if args.pre_train_file else ''
                    warm_up_set = f"--warmup_ratio {args.warmup_ratio}" if args.warmup_ratio else '--warmup_steps 5000'
                    save_steps = 250 if dataset == 'xsum' else 5000
###################### wikitext training part ####################################
# --master_port=12431
#                     command = f"""
# torchrun --nproc_per_node={args.per_node} --master_port=12433 ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --share_freq_across_heads {args.share_freq_across_heads} \\
# --learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
# --dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs {args.num_train_epochs} --num_harmonics {args.num_harmonics} \\
# --eval_strategy steps --eval_steps 250 --logging_dir ./{args.project_name}_log --logging_steps 250 --save_steps {save_steps} --attn_implementation {args.attn_implementation} --path_use_qk_norm false \\
# --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_steps 5000 --path_conv_bias false --output_dir runs/{args.project_name} \\
# --gradient_accumulation_steps {args.gradient_accumulation_steps} --b_unfreeze_step {args.b_unfreeze_step}
# """
# --model_name_or_path
                    command = f"""
torchrun --nproc_per_node={args.per_node} --master_port=12422 ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --share_freq_across_heads {args.share_freq_across_heads} \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --num_train_epochs {args.num_train_epochs} --num_harmonics {args.num_harmonics} \\
--eval_strategy steps --eval_steps 250 --logging_dir ./{args.project_name}_log --logging_steps {args.logging_steps} --save_steps {save_steps} --attn_implementation {args.attn_implementation} --path_use_qk_norm false \\
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 {warm_up_set} --path_conv_bias false --output_dir runs/{args.project_name} \\
--gradient_accumulation_steps {args.gradient_accumulation_steps} --b_unfreeze_step {args.b_unfreeze_step} --pe_method {args.pe_method} --single_A_B {args.single_A_B} \\
--use_beta_modulation {args.use_beta_modulation} --use_soft_wavelet_fox {args.use_soft_wavelet_fox} --wavelet_mode {args.wavelet_mode} {pre_train_file} \\
--wavelet_baseline_use {args.wavelet_baseline_use} --init_theta {args.init_theta} --use_forget_gate {args.use_forget_gate}
"""
###################### wikitext training part ####################################
    if args.debug:
            if args.visualization:
                # max_num = 2000
                step = str(args.steps) + '_' if args.steps != 0 else ''
                command = f"""
    srun -p gpu_intr --gres=gpu:{args.GPU}:1 --pty ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 --model_name_or_path ./{args.project_name}/checkpoint-{max_num} \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --block_size 16384 --dataset_name {dataset} --pivots_num 1 --KL_batch_size 1 \\
    --dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{step}{args.project_name}" --shift {args.shift} \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 2000 --random {random} --random_length {random_length} \\
    --visualization_switch {args.visualization} --visualization_file_name {args.project_name} --alpha {args.alpha} --l2_lambda {args.l2_lambda} --base_select {args.base_select} --report_to none  \\
    --contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none --max_position_embeddings 16384 \\
    --PE_embd_dim {args.PE_embd_dim} --pe_seperate {args.pe_seperate} --use_baseline {args.use_baseline} --transfer_main_ratio {args.transfer_main_ratio} {learn_scale}
    """     
            else:
                block_size = args.block_size if args.block_size != 0 else block_size
                model_name_or_path = f"--model_name_or_path {args.project_name}/checkpoint-{max_num}" if args.steps!=0 else ''
                phase_shift='--phase_shift True' if args.phase_shift else ''
                position_shift='--position_shift True' if args.position_shift else ''
                decay_matrix = '--decay_matrix True' if args.decay_matrix else ''
                scale_reverse = '--scale_reverse True' if args.scale_reverse else ''
                decay_rate = f"--decay_rate {args.decay_rate}" if args.decay_rate != 0 else ''
                model_name_or_path = '--model_name_or_path 18432-512_head_wise_rotary_learn_scale_decay_1w_checkpoint/checkpoint-50000' if dataset == 'xsum' else ""
                command = f"""
    srun -p gpu_intr --gres=gpu:a6000:1 --pty ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 {model_name_or_path} \\
    --learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
    --dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
    --evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 20000 --random {random} --random_length {random_length} \\
    --visualization_switch {args.visualization} --visualization_file_name {args.project_name} --alpha {args.alpha} --l2_lambda {args.l2_lambda} --transfer_main_ratio {args.transfer_main_ratio} \\
    --contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} \\
    --pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
    --attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --report_to none --learn_scale {args.learn_scale} --extra_bias {args.extra_bias} \\
    --freq_base_num {args.freq_base_num} --sample_interval {args.sample_interval} --sample_distribute {args.sample_distribute} {phase_shift} {position_shift} {decay_matrix} {scale_reverse} {decay_rate} \\
    --multiple_w {args.multiple_w} --ln_pos {args.ln_pos} --interpolate_use {args.interpolate_use} --head_wise_rotary {args.head_wise_rotary} --head_wise_decay {args.head_wise_decay} --eta_wise {args.eta_wise} --scale_learn_part {args.scale_learn_part} \\
    --eta {args.eta} --entropy_adaptation {args.entropy_adaptation} --scale_mode {args.scale_mode} --collect_every {args.collect_every} --entropy_apply_mode {args.entropy_apply_mode} \\
    --scale_type {args.scale_type}  --warmup_steps {args.warmup_steps} {model_name_or_path} --gradient_accumulation_steps {args.gradient_accumulation_steps}
    """       
            print('generated command is: \n', command)
            exit()         
    print()
    print()
    print('generated command is: \n', command)
    print()
    print()
    return command
# 要执行的命令
def make_tansorboard_record(args, id):
    # 延迟导入，避免生成脚本阶段因为 torch 环境问题直接崩溃
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        print(f"Warning: failed to import SummaryWriter: {e}. TensorBoard logging disabled.")
        return
    while 1:
        fname = args.project_name + '_checkpoint' if "checkpoint" not in args.project_name else args.project_name
        files_list = glob(f'{fname}/checkpoint*')
        key_word = 'checkpoint'
        max_num = 0
        
        for file in files_list:
                
            sep_file_name = file.split('/')[-1]
            if key_word in sep_file_name:
                num = int(sep_file_name.split('-')[-1])
                if num > max_num:
                    max_num = num
        if args.steps != 0:
            max_num = args.steps
        name = f'{fname}/{key_word}-{max_num}'
        # name = f'{fname}/{key_word}-12500'
        # print(f'Checking {name}')
        # if count == 0:
        #     name = f'{fname}/{key_word}-{39000}'
        # else:
        #     name = f'{fname}/{key_word}-{107550}'
        try:
            os.makedirs(name, exist_ok = True)
            with open(f'{name}/trainer_state.json', 'r') as f:
                trainer_state = json.load(f)
                log_history = trainer_state['log_history']
            jj = fname.split('/')[-1]
            if os.path.isdir(f'runs/{jj}'):
                shutil.rmtree(f'runs/{jj}')
            writer = SummaryWriter(f'runs/{jj}')
            for log in log_history:
                epoch = log.get('epoch', 0.0)  # 确保epoch是浮点数
                if not isinstance(epoch, float):
                    epoch = float(epoch)  # 将epoch转换为浮点数
                epoch_step = epoch * 1000 
                step = log.get('step', 0)
                for key, value in log.items():
                    if key not in ['epoch', 'step']:
                        writer.add_scalar(f'{key}', value, epoch_step)  # 使用浮点数epoch作为横坐标

            # count += 1
            
            print(f'Successfully processed {name}')
            writer.close()
            if args.make_tensorboard:
                exit()
            time.sleep(1200)
            read_process(args, id)
        except:
            time.sleep(1200)
            read_process(args, id)
def generate_parser():
    parser = argparse.ArgumentParser(description="Generate torchrun command based on project name.")
    parser.add_argument('--project_name', type=str, required=True, help='Project name containing parameters.')
    parser.add_argument('--visualization', type=bool, default=False, help='Include visualization flag in the command.')
    parser.add_argument('--test', type=bool, default=False, help='Include test flag in the command.')
    parser.add_argument('--resume', type=bool, default=False, help='Include test flag in the command.')
    parser.add_argument('--GPU', type=str, default='', help='specific GPU requirment')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--num_pre_train', type=bool, default=False, help='dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Include test flag in the command.')
    parser.add_argument('--base_select', type=int, default=10000, help='Include test flag in the command.')
    parser.add_argument('--PE_training_start_step', type=int, default= -1, help='the beginning step of the training of the PE.')
    parser.add_argument('--kl_beta', type=float, default=1.0, help='kl_beta')
    parser.add_argument('--PE_pretrain_file', type=str, default='', help='addr of the PE parameters')
    parser.add_argument('--requires_grad', type=bool, default=False, help='require grads')
    parser.add_argument('--ctt_loss_ratio', type=float, default=0.1, help='beta for contrastive loss')
    parser.add_argument('--per_node', type=int, default=4, help='grab n GPU per node')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode to generate srun instruction')
    parser.add_argument('--get_grad', type=bool, default=False, help='switch to get grad')
    parser.add_argument('--lr_rate', type=str, default='1e-4', help='learning rate')
    parser.add_argument('--model_name_or_path', type=str, default='', help='pre_train_file')
    parser.add_argument('--alpha', type=float, default=1.0, help='exp distribution parameter')
    parser.add_argument('--steps', type=int, default=0, help='mannully setted steps')
    parser.add_argument('--entire_test', type=bool, default=False, help='switch that controls the steps of checkpoints')
    parser.add_argument('--interpolation', type=bool, default=False, help='interpolation')
    parser.add_argument('--PE_freeze_switch', type=bool, default=False, help='switch of the pe freeze')
    parser.add_argument('--hard_negative_switch', type=bool, default=False, help='control the hard negatives')
    parser.add_argument('--shift', type=bool, default=False, help='position random shift switch')
    parser.add_argument('--test_start_step', type=int, default=0, help='ignore the checkpoint before the setted steps')
    parser.add_argument('--test_end_step', type=int, default=10000000000000, help='ignore the checkpoint after the setted steps')
    parser.add_argument('--max_position_embeddings', type=int, default=16384, help='max_position_embeddings for roformer')
    parser.add_argument('--pivots_num', type=int, default=512, help='max_position_embeddings for roformer')
    parser.add_argument('--KL_batch_size', type=int, default = 16, help='max_position_embeddings for roformer')
    parser.add_argument('--warmup_step', type=int, default = 0, help='max_position_embeddings for roformer')
    parser.add_argument('--block_size', type=int, default = 0, help='max_position_embeddings for roformer')
    parser.add_argument('--contrastive_num_list_len', type=int, default = 32, help='contrastive_num_list_len')
    parser.add_argument('--l2_lambda', type=float, default=0.01, help='L2 penalty')
    parser.add_argument('--subtractive_ratio', type=float, default=0.1, help='subtractive ratio')
    parser.add_argument('--relative_attention_num_buckets', type=int, default = 32, help='number of buckets for T5 position encoding')
    parser.add_argument('--make_tensorboard', type=bool, default=False, help='')
    parser.add_argument('--pe_seperate', type=bool, default=False, help='')
    parser.add_argument('--detach', type=bool, default=False, help='')
    parser.add_argument('--learn_scale', type=bool, default=False, help='')
    parser.add_argument('--use_baseline', type=bool, default=False, help='')
    parser.add_argument('--extra_params', type=str, default='', help='')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='')
    parser.add_argument('--pe_method', type=str, required=True, default='none', help='')
    parser.add_argument('--extra_bias', type=str, default='wo', help='')
    parser.add_argument('--resolution', type=int, default=224, help='resolution of the image')
    parser.add_argument('--PE_embd_dim', type=int, default=224, help='pe embedding dimensions')
    parser.add_argument('--dataloader_num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--transfer_main_ratio', type=float, default=0.0, help='transfer_main_ratio')
    ################# freq base related parameters #################################
    parser.add_argument('--sample_interval', type=float, default=0.0001, help='frequency sampling interval of exp function, over large will lead to the nan.')
    parser.add_argument('--freq_base_num', type=int, default=1, help='frequency base number, 1 for the original exp function, 2 for the sin/cos function')
    parser.add_argument('--sample_distribute', type=str, choices=['exp','linear'], default='exp', help='')
    parser.add_argument('--phase_shift', type=bool, default=False, help='')
    parser.add_argument('--position_shift', type=bool, default=False, help='')
    parser.add_argument('--multiple_w', type=str, default='single', choices=['single', 'multiple_dim', 'multiple_layer', 'none'], help='')
    parser.add_argument('--ln_pos', type=str, default='post', choices=['post', 'pre'], help='')
    parser.add_argument('--decay_matrix', type=bool, default=False, help='switch to get grad')
    parser.add_argument('--scale_reverse', type=bool, default=False, help='switch to get grad')
    parser.add_argument('--decay_rate', type=float, default=0.0, help='transfer_main_ratio')
    parser.add_argument('--freq_lr', type=float, default=1e-4, help='frequency learning rate')
    parser.add_argument('--eta', type=float, default=1.0, help='eta for density distortion.')
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help='warmup ratio')
    parser.add_argument('--relative_type', type=int, default=4, help='resolution of the image')
    parser.add_argument('--freq_g_num', type=int, default=1, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='')
    parser.add_argument('--collect_every', type=int, default=250, help='')
    parser.add_argument('--interpolate_use', type=bool, default=False, help='whether use interpolate in testing.')
    parser.add_argument('--get_qk_l2', type=bool, default=False, help='whether to get qk l2.')
    parser.add_argument('--account', type=bool, default=False, help='')
    parser.add_argument('--head_wise_rotary', type=bool, default=False, help='whether to use head-wise rotary embedding.')
    parser.add_argument('--head_wise_decay', type=bool, default=False, help='whether to use head-wise decay.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='', help='')
    parser.add_argument('--scale_mode', type=str, default='unique', help='scale_mode')
    parser.add_argument('--entropy_apply_mode', type=str, default='comp', help='how does eta adapt with entropy change.')
    parser.add_argument('--eta_wise', type=str, default='layer', help='eta wise: layer or head.')
    parser.add_argument('--scale_learn_part', type=str, default='entire', help='learn entire scale or learn exponential part only.')
    parser.add_argument('--scale_type', type=str, default='uniform', help='how the scale of decay term distributes.')
    parser.add_argument('--pre_train_file', type=str, default='', help='pre train file path.')
    parser.add_argument('--decay_apply_type', type=str, default='multiply', help='')
    parser.add_argument('--attn_implementation', type=str, default='path_attn', help='')
    parser.add_argument('--entropy_adaptation', type=bool, default=False, help='whether to use entropy adaptation.')
    parser.add_argument('--share_freq_across_heads', type=bool, default=False, help='whether to use entropy adaptation.')
    parser.add_argument('--single_A_B', type=bool, default=False, help='whether to use entropy adaptation.')
    parser.add_argument('--use_beta_modulation', type=bool, default=False, help='whether use freq in beta.')
    parser.add_argument('--num_harmonics', type=int, default=2, help='')
    parser.add_argument('--logging_steps', type=int, default=250, help='')
    parser.add_argument('--b_unfreeze_step', type=int, default=5000, help='')
    parser.add_argument('--use_soft_wavelet_fox', type=bool, default=False, help='whether to use wavelet-based beta modulation.')
    parser.add_argument('--wavelet_mode', type=str, default='db1', help='wavelet mode for wavelet-based operations.')
    parser.add_argument('--wavelet_baseline_use', type=bool, default=False, help='whether to use wavelet-based beta modulation.')
    parser.add_argument('--init_theta', type=float, default=0.847, help='initial theta for path attention ratio')
    parser.add_argument('--use_forget_gate', type=bool, default=False, help='whether to use forget gate.')
    
    ################# freq base related parameters #################################

    return parser.parse_args()


def get_valuable_GPU(args):

    command = "bash GPU_info.sh"


    # 执行命令并获取返回结果
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    data = result.stdout.split('\n')
    result = {}
    # 遍历数据列表
    for item in data:
        
        # 检查是否包含a6000或a100
        if args.GPU:
            if args.GPU in item:
                # 使用split()方法获取需要的部分
                parts = item.split()
                if parts:
                    # 提取第一个部分（比如elm72）
                    result[parts[0]] = parts[1]            
        else:
            parts = []
            # if 'a6000' in item or 'a100' in item or '6000' in item or 'v100' in item:
            if args.test:
                if ('a6000' in item or 'a100' in item or '6000' in item):
                    parts = item.split()
                    if parts:
                        # 提取第一个部分（比如elm72）
                        result[parts[0]] = parts[1]                    
            else:
                if ('a6000' in item or 'a100' in item or '6000' in item):
                # if 'a6000' in item:
                    # 使用split()方法获取需要的部分
                    parts = item.split()
                    if parts:
                        # 提取第一个部分（比如elm72）
                        result[parts[0]] = parts[1]
            # if len(parts) == 0:
            #     if 'a6000' in item or 'a100' in item or '6000' in item or '3090' in item or 'v100' in item:
            #         parts = item.split()
            #         if parts:
            #             # 提取第一个部分（比如elm72）
            #             result[parts[0]] = parts[1]                    
            # pdb.set_trace()
            # try:
            #     if not parts:
            #         pass
            # except:

            #     if '3090' in item or 'v100' in item:
            #         parts = item.split()
            #         if parts:
            #             # 提取第一个部分（比如elm72）
            #             result[parts[0]] = parts[1]
    return result

def run_bash_file(bash_file):
    result = subprocess.run(f"sbatch {bash_file}", shell=True, capture_output=True, text=True)
    # pdb.set_trace()
    print(result)
    id = result.stdout.split()[-1]
    print(result.stdout)
    return id
# pdb.set_trace()
args = generate_parser()
# 注意：id 在提交作业后才有，不能在这里调用 make_tansorboard_record

GPU_dict = get_valuable_GPU(args)
gress = None
account_text = ''
partion = 'gpu_long'
for key, value in GPU_dict.items():
    if 'lang' in key:
        partion = 'lang_gpu_long'
        account_text = '#SBATCH --account=lang'
        # account_text = '--account lang'
    try:
        if int(value.split(':')[-1]) < args.per_node:
            print(f'{value.split(":")[-1]} valuable GPU is less than your expectation!!!!!!!! ')
    except:
        gress = 'gpu:6000:4'
    gress = value
    if args.GPU == 'a100' and 'a100-80' in value:
        continue
    # node = key   
    break

if gress is None:
    gress = "gpu:a6000:4"
gress = gress.split('(')[0]
gress = gress[:-1] 
gress = gress + '4'
if args.test:
    show_result_in_excel_form = 'python ./log_file/info_extraction.py --id {}'
command = generate_torchrun_command(args, id, account_text)

initial_start_step = f'_start_from_{args.test_start_step}' if args.test_start_step else ''
specific_step = f'_step{args.steps}' if args.steps else ''
record_name = f"{args.project_name}{initial_start_step}{specific_step}_test" if args.test else f"{args.project_name}"
# 避免日志文件名中包含斜杠，导致 Slurm 无法创建输出文件
safe_record_name = record_name.replace('/', '_')
ID = '${SLURM_JOB_ID}'
record_command = f"python log_file/info_extraction.py --project_name {record_name}_{ID}" if args.test else ""
# gress = 'gpu:6000:3'
test_folder = 'test' if args.test else 'train'
# if '(' in gress:
#     gress = 'gpu:6000:4'
# gress = 'gpu:a100:4'
partion = 'lang_gpu_long' if args.account else 'gpu_long'

account_text = '#SBATCH --account=lang' if args.account else ''
#SBATCH --nodelist=elm64
# 优先使用自动探测到的 GRES；如未探测到则回退到单卡 a6000
device_use = f"gpu:{args.GPU}:{args.per_node}"
content = f"""#! /bin/bash
#SBATCH --job-name=
#SBATCH --output=log_file/{test_folder}/%j_{safe_record_name}.txt
#SBATCH --partition={partion}
#SBATCH --time=100:00:00
#SBATCH --gres={device_use}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
{account_text}


export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# 打印调试信息
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
"""

# 打开文件并写入内容
# 1) 确保日志目录存在，避免 Slurm 无法创建输出文件导致脚本静默退出
os.makedirs('log_file/test', exist_ok=True)
os.makedirs('log_file/train', exist_ok=True)

# 2) 生成的训练/评测命令必须真正写入到 sbatch 脚本里，否则作业什么也不会跑
#    同时将工作目录固定为当前目录，增加一些调试输出
workdir = os.getcwd()
args.project_name = args.project_name.split('/')[-1]
train_test_sign = 'test' if args.test else 'train'
bash_file = f"{train_test_sign}_{args.project_name}.sh"
with open(bash_file, "w") as file:
    file.write(content)
    file.write(f"\nset -euxo pipefail\n")
    file.write(f"echo 'Workdir: {workdir}'\n")
    file.write(f"cd {workdir}\n")
    # 把 Python 生成的 torchrun 命令附加到脚本中执行
    file.write("echo 'Launching training/eval command'\n")
    file.write("echo '================= BEGIN RUN ================='\n")
    file.write("\n")
    file.write(command)
    file.write("\n")

# 3) 提交作业
id = run_bash_file(bash_file)
# 4) 训练情况下，持续把 trainer_state.json 写入 TensorBoard（测试模式不需要）
# if not args.test:
#     make_tansorboard_record(args, id)