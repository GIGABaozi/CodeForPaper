# Auxiliary Loss for Instruction-Finetuning

This repository provides a drop-in modification to the Open-Instruct finetuning script, adding our proposed auxiliary loss.  

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
