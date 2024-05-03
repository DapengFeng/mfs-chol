#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_2=build/examples/test_elasticity

deform_mesh_repo=("dat/bmw-cage-compress.off" "dat/bmw-cage-bend.off" "dat/bmw-cage-twist.off")

query_mesh=dat/bmw.off
mesh=dat/bmw-cage-rest.off

pcg_iters=10
nu=0.45
rho=3
eps=0.001

rm *kelvin_3d*.bin

for deform_mesh in "${deform_mesh_repo[@]}"; do
    
filename=$(basename "${deform_mesh}")
name="${filename%.*}"
echo "${name}"

outdir=result/kl_res/kelvin/${name}
mkdir -p ${outdir}

${exe_2} source_mesh=${mesh} target_mesh=${query_mesh} deformed_source_mesh=${deform_mesh} outdir=${outdir} pcg_iters=${pcg_iters} nu=${nu} eps=${eps} rho=${rho} sec_num=16 2>&1 | tee ${outdir}/log-${name}-pcg-${pcg_iters}.txt

done
