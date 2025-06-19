#!/bin/bash
#BSUB -J labram_train
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -q gpuv100                 
#BSUB -W 4:00                     # Increasing time to four hours
#BSUB -n 8                        # Request 8 CPU cores
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 full GPU
# Load Python 
module load python3/3.10.16

# Activate virtual environment
source /zhome/4d/f/205277/labram_env/bin/activate

# Run training script
python3 scripts/train_v2.py
