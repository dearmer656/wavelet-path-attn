#! /bin/bash
#SBATCH --job-name=owt_download
#SBATCH --output=log_file/train/%j_owt_download.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

set -euxo pipefail
echo 'Downloading and caching OpenWebText dataset'
python3 -c "
from datasets import load_dataset
import os
cache_dir = '/project/nlp-work5/hongyu-s/huggingface/datasets'
print(f'Downloading openwebtext to {cache_dir}')
ds = load_dataset('openwebtext', cache_dir=cache_dir, trust_remote_code=True)
print(f'Done. Train size: {len(ds[\"train\"])} examples')
"
echo 'Download complete.'
