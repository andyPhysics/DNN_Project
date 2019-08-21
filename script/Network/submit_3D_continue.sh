#PBS -A PAS1495

#PBS -N Neural_loss_continuedx

#PBS -l walltime=72:00:00

#PBS -l nodes=1:ppn=40

#PBS -j oe

# uncomment if using qsub

cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

module load python/2.7-conda5.2

python -u /users/PAS1495/amedina/work/DNN_Project/script/Network/Continue_test.py >&output_3D_continued new_3D_continued.log
