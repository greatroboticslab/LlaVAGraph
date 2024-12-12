# 📊 LLaVAGraph


## Contents
- [Install](#install)
- [Finetuning](#Finetuning)
- [Demo](#Demo)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

### Slurm Install

```bash
git clone https://github.com/greatroboticslab/LlaVAGraph/
cd LLaVAGraph
```

If you're running this on the MTSU cluster, this shouldn't be too bad... Modify `slurm/dependencies.sbatch` with your desired paths and then run it.

```bash
#!/bin/bash
#SBATCH --job-name=file-test     # Job name
#SBATCH --partition=a100         # Partition (queue) name
#SBATCH --gres=gpu:A100:1        # Request 1 A100 GPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks (usually 1 for GPU jobs)
#SBATCH --time=00:30:00          # Time limit (hh:mm:ss)
#SBATCH --output=my_job.out      # Standard output
#SBATCH --error=my_job.err       # Standard error

# Load any necessary modules or environments
echo "==> Loading modules"
module load cuda/12.4
echo "==> Creating virtual environment"
python -m venv /projects/<your-username>/llava
echo "==> Activating conda environment"
source /projects/<your-username>/llava

echo "==> Activating conda environment"
source /projects/imo2d/llava
echo "==> Installing torch"
pip install torch
pip install --upgrade pip  # enable PEP 660 support
echo "==> Installing LLaVA"
pip install -e /home/<your-directory>/LLaVA/ 
echo "==> Installing LLaVA-Train"
pip install -e "/home/<your-directory>/LLaVA[train]"
echo "==> Installing flash-attn"
pip install flash-attn --no-build-isolation
echo "==> Installing deepspeed"
pip install deepspeed
echo "==> Done!"
```

```
sbatch slurm/dependencies.sbatch
```

### Manual Install

Hopefully you won't have to do this, but these instructions are here for reference.

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. Download LLaVA weights
```Shell
bash ./download-llava.bash <save-dir>
```

5. Install `deepspeed`
```Shell
pip install deepspeed
```


## Finetuning

### Dataset Format

Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), and `conversations` (the conversation data between human and AI).

A sample JSON for finetuning LLaVA for generating tag-style captions for Stable Diffusion:

```json
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
```

### Modifying Training Parameters

```bash

deepspeed  <path-to-llava>/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed <path-to-llava>/scripts/zero3.json \
    --model_name_or_path <path-you-saved-the-model> \
    --version v1 \
    --data_path <where-you-saved-the-images>/trainingData.json \
    --image_folder <where-you-saved-the-images> \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir <where-you-want-to-save-checkpoints> \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \ 
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \ 
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \ 
    --report_to none 
```

Once you get this setup correctly, you should be able to just run:

```
sbatch slurm/training.sbatch
```

And get your final output.

## Evaluation



## Acknowledgements

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- [LLaVA](https://github.com/haotian-liu/LLaVA): the base for our models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

