#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -J conf
#SBATCH --partition=barnard
#SBATCH --output=crst.out
#SBATCH --error=crst.err
#SBATCH -A p_phononics
#SBATCH --ntasks-per-node=104
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3900M

ulimit -s unlimited

# load Anaconda and the environment created for running crest
module  load Anaconda3/2023.07-2
source activate /home/medranos/.conda/envs/crest

# replace with path to crest executable
chmod +x /data/horse/ws/medranos-proj2023/alejandra/code/crest
export PATH=$PATH:/data/horse/ws/medranos-proj2023/alejandra/code

# molecule id whose conformational search will be executed
id=0
# path where all output files from crest will be stored
mainDir=/data/horse/ws/medranos-proj2023/alejandra/data/$id
mkdir $mainDir
cd $mainDir

# path where id.xyz file is stored
mol=/data/horse/ws/medranos-proj2023/alejandra/data/$id.xyz
# small molecules
crest $mol -gfn2 -gbsa h2o -mrest 10 -rthr 0.1 -ewin 12.0 -T 104
# medium-size molecules
crest $mol -gfn2 -gbsa h2o -opt normal -quick -mrest 5 -rthr 0.1 -ewin 12.0 -T 104
# large molecules
crest $mol -gfn2 -gbsa h2o -opt lax -norotmd -mquick -mrest 5 -rthr 0.1 -ewin 12.0 -T 104