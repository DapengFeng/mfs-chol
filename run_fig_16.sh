#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe_0=build/examples/test_diffusion_meshes

model_name=demosthenes

source_mesh=dat/${model_name}/${model_name}.off
source_charge=dat/${model_name}/${model_name}.dat
target_mesh=dat/${model_name}/${model_name}-plane.off

outdir=result/kl_res/diffusion_meshes/${model_name}/

filename=$(basename "${source_mesh}")
name="${filename%.*}"
echo ${name}

eps=0.0001
wave_num=40

mkdir -p ${outdir}

rm *laplace_3d*.bin

for rho in 2 4 6 8; do
frm=`echo "${rho}/2+1" | bc`
out_mesh=${outdir}/mesh-rho-${frm}.obj
out_data=${outdir}/data-rho-${frm}.dat

${exe_0} outdir=${outdir} source_mesh=${source_mesh} source_charge=${source_charge} target_mesh=${target_mesh} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=10000000 pcg_iters=0 out_mesh=${out_mesh} out_data=${out_data} use_pcg=false reset_charge=true wave_num=${wave_num} 2>&1 | tee ${outdir}/log-${name}-rho-${rho}.txt
done

pcg_iters=32
out_mesh=${outdir}/mesh-rho-1.obj
out_data=${outdir}/data-rho-1.dat

${exe_0} outdir=${outdir} source_mesh=${source_mesh} source_charge=${source_charge} target_mesh=${target_mesh} num_sec=16 eps=${eps} rho=6 max_supernode_size=10000000 pcg_iters=${pcg_iters} out_mesh=${out_mesh} out_data=${out_data} use_pcg=true reset_charge=true wave_num=${wave_num} 2>&1 | tee ${outdir}/log-${name}-rho-inf.txt
