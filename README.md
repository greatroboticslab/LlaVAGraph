# 🌋📊 LLaVAGraph


## Contents
- [Install](#install)
- [Finetuning](#Finetuning)
- [Evaluation](#Evaluation)
- [Acknowledgements](#Acknowledgements)


## Install

2. Install Package
```Shell
python -m venv /projects/<username>/llava
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

### Installation

Currently, evaluation requires a separate virtual environment for running LLAMA 3.2 3B (<https://huggingface.co/meta-llama/Llama-3.2-3B>). You'll need to request access to those models through Huggingface first (it took me less than an hour to get it approved, but your mileage may vary...)

```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --exclude "original/*" --local-dir Llama-3.2-3B-Instruct
```

<details>
<summary>MTSU cluster users</summary>
  
```
mv Llama-3.2-3B-Instruct /projects/<username>/Llama-3.2-3B-Instruct
```
  
(For whatever reason, this doesn't work well if you set `local-dir` to include the `/projects/` directory, so you'll need the extra step)
  
</details>

### Running Captioning

Look at the paths in `eval/evaluateLLaVA.sh` and change them to fit your current folder setup. Then run the training script:

```
bash scripts/evaluateLLaVA.sh
```

This will create three files in `eval/results/llava` - `randomNoise.json`, `sineWave.json`, and `squareWave.json`. This generates a description of the image:

```json
{
    "image": "SquareTrials-3-100Hz-100Hz.xlsx-17.png",
    "conversation": [
      {
        "question": "Is the line shown in the graph continuous? Describe the line.",
        "answer": "<s> This wave exhibits a non-random, yet discontinuous, pattern with sudden shifts to symmetrical peak excursions.</s>"
      },
      {
        "question": "Does the graph contain any random points?",
        "answer": "<s> The continuous line's transitions between two distinct levels are regular and predictable, demonstrating a deterministic process.</s>"
      },
      {
        "question": "Does the graph contain sharp corners?",
        "answer": "<s>  While non-random, this graph exhibits sharp corners and abrupt decreases in value.</s>"
      }
    ]
  },
```

Once these descriptions are generated, we can now run `eval/categorizeLLAMA.sh`, which takes the following information and makes a decision.

```json
{
    "conversationId": "SquareTrials-3-100Hz-100Hz.xlsx-17.png",
    "response": "Based on the information provided, I would select option C) Square wave.\n\nThe reasoning behind this decision is as follows:\n\n1. The line exhibits sharp corners and abrupt decreases in value, which aligns with the characteristics of a square wave.\n2. The line is non-random, yet discontinuous, which is also consistent with the properties of a square wave.\n3. The line's transitions between two distinct levels are regular and predictable, indicating a deterministic process, which is another characteristic of square waves.\n4. The line does not exhibit gradual transitions or easily discernible structure, which distinguishes it from a sine wave.\n\nThese characteristics collectively support the conclusion that the graph represents a square wave."
},
```

## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA): the base for our models
  - [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- [LLAMA 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B): our reasoning model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

