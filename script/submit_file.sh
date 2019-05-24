#PBS -A PAS1495

#PBS -N Make_plots

#PBS -l walltime=0:30:00

#PBS -l nodes=1:ppn=40:gpus=1

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/3.6-conda5.2

module load cuda/10.0.130

python -u make_plots.py >&output_plots make_plots.log
