#!/bin/bash

for scale in 1 2 4 8 16 32 64
  do
    python performance_invert_ng_large.py --scale ${scale} --nthreads 64
  done
