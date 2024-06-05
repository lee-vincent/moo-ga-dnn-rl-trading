#!/bin/bash

N_GEN=25
N_POP=100
TICKER="tqqq"
JOB_START_TIME="$(TZ='America/New_York' date +'%m-%d-%Y_%I%M%p')"
FNCT="Tanh"
FNCR="ReLu"

for i in {1..25}; do
    nohup python3 -u main.py --n_gen $N_GEN --pop_size $N_POP --ticker $TICKER --fnc $FNCT --force_cpu > "/tmp/${TICKER}_ngen-${N_GEN}_npop-${N_POP}_${JOB_START_TIME}_${FNCT}_${i}.txt" 2>&1 &
done

for i in {1..25}; do
    nohup python3 -u main.py --n_gen $N_GEN --pop_size $N_POP --ticker $TICKER --fnc $FNCR --force_cpu > "/tmp/${TICKER}_ngen-${N_GEN}_npop-${N_POP}_${JOB_START_TIME}_${FNCR}_${i}.txt" 2>&1 &
done