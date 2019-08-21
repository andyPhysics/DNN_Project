#PBS -A PAS1495

#PBS -N Neural_loss_huge

#PBS -l walltime=120:00:00

#PBS -l nodes=1:ppn=20,mem=600GB

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/2.7-conda5.2

python -u /users/PAS1495/amedina/work/DNN_Project/script/Network/New_3D.py >&output_3D_huge new_3D_huge.log
