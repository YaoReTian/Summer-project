#!/bin/bash
#PBS -lwalltime=05:00:00
#PBS -lselect=1:ncpus=1:mem=16gb
#PBS -N bpskPipeline

module load SciPy-bundle/2025.07-iimkl-2025b
module load Python/3.13.5-GCCcore-14.3.0

cd ${PBS_O_WORKDIR}/modules

python3 test.py

