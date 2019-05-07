#!/bin/bash

#SBATCH --account=scw1000  # FIXME, likely scw1469

# Output file: CNNLSTM_AE_HPC.<job number>.out
#SBATCH -o CNNLSTM_AE_HPC.%J.out

# Error messages output file: CNNLSTM_AE_HPC.<job number>.err
# This contains Tensorflow Diagnostic messsages.
#SBATCH -e CNNLSTM_AE_HPC.%J.err

#SBATCH --partition=gpu

# We require ONE GPU.
#SBATCH --gres=gpu:1        

#SBATCH --ntasks=1         

#20GB should be much more than needed, but should also be available.
#SBATCH --mem=20G          

# For the following, we use '&&' to concatenate commands
# so that if one fails the following ones are not executed.

DATASIZE=$1
EPOCHS=$2
BATCHSIZE=$3

module purge && # we remove all modules for safety - we only need the tensorflow related modules.
module load tensorflow/1.11-gpu  &&
date  &&
mkdir -p RESULTS &&
echo "Creating fake data" &&
python3 ./create_fake_data.py $DATASIZE &&
echo "Created fake data" && 
echo "Running TF program" &&
time python3 ./CNNLSTM_AE_HPC.py fakedata.csv $EPOCHS $BATCHSIZE && # CALL YOUR SCRIPT HERE.
echo "Run TF program" &&
date &&
echo "Bye"
