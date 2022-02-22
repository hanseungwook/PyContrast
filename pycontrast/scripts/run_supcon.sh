#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/PyContrast-SupCon/pycontrast/launcher.py \
      --method MoCov2 \
      --cosine \
      --data_folder /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --multiprocessing-distributed \
      --world-size 3 \
      --rank 0 \
      --num_workers 32 \
      --ngpus 6 \
      --model_path $HOME2/scratch/PyContrast-SupCon/pycontrast/saved_models/ \
      --tb_path $HOME2/scratch/PyContrast-SupCon/pycontrast/logs/ \
      --sup_mode supcon

echo "Run completed at:- "
date

