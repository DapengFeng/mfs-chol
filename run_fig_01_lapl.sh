#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_0=build/examples/test_diffusion_meshes

model_name=tower

# from 001 to 240
frame=004

source_mesh=dat/teaser-lap-tower.off
source_charge=dat/tower-teaser/tower-temperature-${frame}.dat
target_mesh=dat/teaser-lap-query.off

outdir=result/kl_res/teaser_tower_meshes/
out_mesh=${outdir}/mesh-${frame}.obj
out_data=${outdir}/data-${frame}.dat
echo "${out_mesh}"

filename=$(basename "${source_mesh}")
name="${filename%.*}"
echo ${name}

eps=0.0001
rho=5.5
pcg_iters=7

mkdir -p ${outdir}

rm *laplace_3d*.bin

${exe_0} outdir=${outdir} source_mesh=${source_mesh} source_charge=${source_charge} target_mesh=${target_mesh} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=10000000 pcg_iters=${pcg_iters} out_mesh=${out_mesh} out_data=${out_data} use_gui=false revert_transform=true 2>&1 | tee ${outdir}/log-${name}.txt
