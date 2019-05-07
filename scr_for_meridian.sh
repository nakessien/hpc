#!/bin/bash

SBATCH=sbatch
SQUEUE=squeue
REALDATA=clean_bm19.csv
USERNAME=s.a.e.essien

for DATASIZE in 67500 125000 250000 500000
do
  for BATCHSIZE in 5 20 80 320 1280 5120
  do 
    # This is to compute a number of epochs that would run in less than a minute
    TIMECONSTANTGPU=0.014 # minimum measured time per batch per epoch on GPUS (seconds)
                          # this does not basically change if the batch size is below 500
    NBATCHES=$((DATASIZE/BATCHSIZE))
    TIME=60 # seconds
    EPOCHS=$(echo | awk '{print int('$TIME'/'$TIMECONSTANTGPU'/'$NBATCHES'/4)}')
    # end computation of epochs
    echo $DATASIZE $EPOCHS $BATCHSIZE $TIME
    # Run without SLURM submission:
    #./slurm_batch.sh $DATASIZE $EPOCHS $BATCHSIZE fake 
    #
    # actual job submission to slurm.
    JOBID=$($SBATCH -t 10 ./slurm_batch.sh $DATASIZE $EPOCHS $BATCHSIZE $REALDATA | cut -d' ' -f4 )
    echo $JOBID was submitted

    # To wait for the current job to finish - as output files may conflict.
    # if you don't want to wait, you can create other directories to run in
    # and put in them what is needed.
    while ($SQUEUE -u $USERNAME -O JobID | grep $JOBID &> /dev/null ) 
    do 
      echo waiting on $JOBID
      sleep 3
    done
    # wait ended
  done
done

wait
