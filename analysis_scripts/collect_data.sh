#!/bin/bash
JOBIDS=$(for file in *.err ; do f=${file#*HPC.}; echo ${f%.err}; done)

echo -e "JobID\tRealTime\tTrSize\tEpochs\tBtchSize"
for jobid in $JOBIDS
do 
  echo $jobid $( grep real CNNLSTM_AE_HPC.$jobid.err | sed 's/\r/ /g') $(grep Train CNNLSTM_AE_HPC.$jobid.out | sed 's/\r/ /g')
done | sed -E 's/([0-9]{5})\s+real\s+([0-9]+m[0-9]{1,2}\.[0-9]+s).*Train\s+shape\:\s+\(([0-9]+).*\).*\s+([0-9]+)\s+epochs,\s+([0-9]+)\sbatch\ssize/\1\t\2\t\3\t\4\t\5/'

mkdir -p LOSS_FUNCTIONS
for file in *.out
do
   JOBID=$(f=${file#*HPC.}; echo ${f%.err})
   OUTPUT=LOSS_FUNCTIONS/$JOBID.tsv
   echo -e "Duration(s)\tLoss" > $OUTPUT
   grep loss $file | sed -E 's/.*\s+([0-9]+)s\s+.*loss:\s*([0-9.]*).*$/\1\t\2/' >> $OUTPUT
done







#for jobid in $JOBIDS
#do 
#  echo $jobid $( grep real CNNLSTM_AE_HPC.$jobid.err | sed 's/\r/ /g') $(grep Train CNNLSTM_AE_HPC.$jobid.out | sed 's/\r/ /g')
#done
