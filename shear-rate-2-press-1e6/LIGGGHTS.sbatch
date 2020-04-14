#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p hpxg
#SBATCH --job-name=sr-2
module load openmpi
export LD_LIBRARY_PATH=/software/VTK/7.1.1_gcc7.3_openmpi3.1.3/lib:$LD_LIBRARY_PATH
srun --mpi=pmi2 -p hpxg -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.shear -var try 1 -var norm_press 1e6 -var shear_rate 2 -var delta_strain_dump 0.001 -var delta_strain_output 1e-5 -var shear_strain 5.0
#srun --mpi=pmi2 -p hpxg -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.shear -var try 2 -var norm_press 1e6 -var shear_rate 2 -var delta_strain_dump 0.001 -var delta_strain_output 1e-5 -var shear_strain 5.0
#srun --mpi=pmi2 -p hpxg -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.shear -var try 3 -var norm_press 1e6 -var shear_rate 2 -var delta_strain_dump 0.001 -var delta_strain_output 1e-5 -var shear_strain 5.0
#srun --mpi=pmi2 -p hpxg -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.shear -var try 4 -var norm_press 1e6 -var shear_rate 2 -var delta_strain_dump 0.001 -var delta_strain_output 1e-5 -var shear_strain 5.0
#srun --mpi=pmi2 -p hpxg -n 16 /home/magang/project/software/LIGGGHTS-PUBLIC-3.1.0+domain/src/lmp_fedora<in.shear -var try 5 -var norm_press 1e6 -var shear_rate 2 -var delta_strain_dump 0.001 -var delta_strain_output 1e-5 -var shear_strain 5.0