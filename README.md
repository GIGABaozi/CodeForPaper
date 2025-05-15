# Auxiliary Loss for Instruction-Finetuning

This repository provides a drop-in modification to the Open-Instruct finetuning scripts, adding our proposed auxiliary loss.

## Contents

- `finetune_new.py`  
  Modified main training script: replace the original `finetune.py` with this file to enable the auxiliary loss.  
- `finetune_aux.py`  
  Defines the auxiliary‐loss functions and any supporting utilities.  

All other code and folder structure remain exactly as in the base Open-Instruct release.

## Installation

1. Clone or download the Open-Instruct repo.  
2. Copy our two files into the folder:
   ```bash
   cp finetune_new.py \
      finetune_aux.py \
      /root/.jupyter/lab/workspaces/open-instruct-main/open_instruct/
   ```
3. Rename the original script:
   ```bash
   mv /root/.jupyter/lab/workspaces/open-instruct-main/open_instruct/finetune.py \
      /root/.jupyter/lab/workspaces/open-instruct-main/open_instruct/finetune_orig.py
   ```
4. Rename our main script:
   ```bash
   mv /root/.jupyter/lab/workspaces/open-instruct-main/open_instruct/finetune_new.py \
      /root/.jupyter/lab/workspaces/open-instruct-main/open_instruct/finetune.py
   ```

## Enabling Auxiliary Loss in Tulu Mix Script

Edit `open-instruct-main/scripts/train/finetune/tulu_finetune_mix.sh` to enable the auxiliary loss. Locate the training command and add:

```bash
--load_balancing_weight $weight \
--load_balancing_loss True
```

Here, `$weight` corresponds to the “γ” (gemma) parameter in the paper.

## Usage

Run finetuning as before. The modified `finetune.py` will automatically import and apply the auxiliary loss defined in `finetune_aux.py` alongside the standard instruction-tuning loss, and the mix script will include the new flags.

```bash
bash scripts/train/finetune/tulu_finetune_mix.sh --load_balancing_weight 50 --load_balancing_loss True
```

Everything else (data paths, hyperparameters, logging) is unchanged.


