#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p hpib
SLURM_JOB_NAME=press-1e6
module load openmpi
export LD_LIBRARY_PATH=/software/VTK/7.1.1_gcc7.3_openmpi3.1.3/lib:$LD_LIBRARY_PATH
srun --mpi=pmi2 -p hpib -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.sample -var try 1 -var insertname in.particles-1 -var fric 0.1 -var norm_press 1e6
#srun --mpi=pmi2 -p hpib -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.sample -var try 2 -var insertname in.particles-2 -var fric 0.1 -var norm_press 1e6
