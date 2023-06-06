#!/bin/bash

: ${NODES:=1}

srun -N $NODES --exclusive --partition=shpc          \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  /usr/local/cuda/bin/ncu --set full ./translator $@
