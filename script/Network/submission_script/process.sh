#PBS -A PAS1495

#PBS -N Image_processing

#PBS -l walltime=24:00:00

#PBS -l nodes=1:ppn=40:gpus=2

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/2.7.5

module load cuda/10.0.130

python -u image_processing.py >&image_output image_output.log
