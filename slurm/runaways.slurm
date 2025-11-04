#!/bin/bash
## Job Name
#SBATCH --job-name=runaway-defs
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3:00:00
#SBATCH -o /mnt/home/twagg/projects/runaway-definitions/slurm/logs_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# choose velocity dispersion from array based on slurm task id
disp_array=(0.1 2 2)
kickflag_array=(5 5 1)
disp=${disp_array[$SLURM_ARRAY_TASK_ID]}
kickflag=${kickflag_array[$SLURM_ARRAY_TASK_ID]}

echo "Generating runaway population with dispersion ${disp} km/s and kick flag ${kickflag}"

output_path=/mnt/home/twagg/ceph/runaway-defs/disp_${disp}_r_3_kick_${kickflag}.h5

echo "Output path: ${output_path}"

python /mnt/home/twagg/projects/runaway-definitions/src/generate_runaway_pop.py -d $disp -o $output_path -r 3 -k $kickflag -n 100 -c $SLURM_NTASKS_PER_NODE
