#!/bin/bash
#PBS -lwalltime=05:00:00
#PBS -lselect=1:ncpus=1:mem=16
#PBS -N bpskPipeline

module load Python/3.12.3-GCCcore-13.3.0

cd ${PBS_O_WORKDIR}

python3 modules/test.py

