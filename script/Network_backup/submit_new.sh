#PBS -A PAS1495

#PBS -N Neural_regression

#PBS -l walltime=24:00:00

#PBS -l nodes=4:ppn=40:gpus=2

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/2.7-conda5.2

module load cuda/10.0.130

python -u New_network.py >&output_regression5 Neural_regression_new.log
