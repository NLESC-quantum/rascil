#!/usr/bin/env bash
results_dir=/alaska/vlad/data/ska_mid_simulations_results
for duration in long
  do
    for dec in +15 -15 -45 -75
      do
        python3 mid_simulation.py --mode heterogeneous --use_slurm False --flux_limit 0.01 \
          --declination ${dec} --band B2 --results ${results_dir} --rmax 1e4 \
          --nworkers 8 --processes 1 --nthreads 1 --cores 16 --memory 512GB \
          --image_pol stokesIQUV --vis_pol linear --duration ${duration} --npixel 8192 --imaging_context None \
          --configuration MID
      done
  done

