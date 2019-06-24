#!/bin/bash
module load python/3.6
python3 -u train.py MRCONSO_Train_Length_30.txt MRCONSO_Validate_Length_30.txt List_of_Unique_Atoms_Length_30.txt bio_embedding_intrinsic.txt >> output.log

#python -u l-jaccard.py MRCONSO_Parsed_Length_30.txt >> output.log

# sbatch --partition=gpu --gres=gpu:v100:2 --mem=50g --time=24:00:00 git-submit-job.sh
# sbatch --partition=norm --cpus-per-task=2048 --mem=200g --time=10-00:00:00 script.sh
