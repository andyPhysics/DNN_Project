#PBS -A PAS1495

#PBS -N Neural_loss_simple_down

#PBS -l walltime=65:00:00

#PBS -l nodes=1:ppn=20,mem=400GB

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/2.7-conda5.2

python -u /users/PAS1495/amedina/work/DNN_Project/script/Network/New_simple_down.py >&output_simple_down new_3D_simple_down.log
