#!/bin/bash

# Run this in a screen. (google for 'gun screen')

echo 0 > JOBIDs

sbatchmock(){

  JOBID=$(tail -n 1 JOBIDs)
  JOBID=$((JOBID+1))
  echo "Submitted job id $JOBID"
  echo $JOBID >> JOBIDs

}

export sbatchmock

SBATCH=sbatch

DEPSTRING=""
for DATASIZE in 100000 200000 300000 
do
  for EPOCHS in 100 200 400
  do 
    for BATCHSIZE in 50 100 200 400
    do 
      TIMECONSTANTGPU=0.08 # max time measured per batch on CPUS
      TIME=$(echo | awk '{print int('2*$DATASIZE'/'$BATCHSIZE'*'$EPOCHS'*'$TIMECONSTANTGPU'/60+2)}')
      echo $DATASIZE $EPOCHS $BATCHSIZE $TIME
      #echo DEPSTRING:$DEPSTRING 
      #DEPSTRING="--dependency=afterok:$($SBATCH -t $TIME $DEPSTRING ./slurm_batch.sh $DATASIZE $EPOCHS $BATCHSIZE | cut -d' ' -f4 )"
      #if ! [ "$DEPSTRING" ] 
      #then
      #  exit 1
      #fi
      JOBID=$($SBATCH -t $TIME $DEPSTRING --qos=highpriority ./slurm_batch.sh $DATASIZE $EPOCHS $BATCHSIZE fake | cut -d' ' -f4 )
      echo $JOBID was submitted
      while ( squeue -u s.michele.mesiti -O JobID | grep $JOBID &> /dev/null ) 
      do 
        echo waiting on $JOBID
        sleep 30
      done
    done
  done
done
