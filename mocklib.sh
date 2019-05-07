#!/bin/bash

echo 0 > JOBIDs
sbatchmock(){

  JOBID=$(tail -n 1 JOBIDs)
  JOBID=$((JOBID+1))
  echo "Submitted job id $JOBID"
  echo $JOBID >> JOBIDs
  echo $JOBID >> WORKING

}

squeuemock(){
 touch WORKING
 cat WORKING
}

slurmmock(){

  while true
  do
  sleep 10
  rm -f WORKING
  done

}

export sbatchmock
export squeuemock

slurmmock &


