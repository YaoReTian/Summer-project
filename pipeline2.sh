#!/bin/bash
#PBS -lwalltime=05:00:00
#PBS -lselect=1:ncpus=1:mem=16gb
#PBS -N ldpcPipeline

module load Pillow/11.3.0-GCCcore-14.3.0
module load SciPy-bundle/2025.07-iimkl-2025b
module load Python/3.13.5-GCCcore-14.3.0
module load tools/prod

cd ${PBS_O_WORKDIR}/modules

virtualenv venv
source venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade pyldpc
python3 noisepipeline2withldpc.py

deactivate
