#!/bin/bash
#BSUB -J EEGPreprocess
#BSUB -q hpc
#BSUB -o logs/preprocess_%J.out
#BSUB -e logs/preprocess_%J.err
#BSUB -q hpc                  
#BSUB -W 4:00                     # Increase time to four hours
#BSUB -n 8                        # Request 8 CPU cores 
#BSUB -R "rusage[mem=8GB]"

# Load Python module
module load python3/3.10.16

# Activate virtual environment
source /zhome/4d/f/205277/labram_env/bin/activate

# Run the preprocessing script
python3 scripts/process_data_v2.py
