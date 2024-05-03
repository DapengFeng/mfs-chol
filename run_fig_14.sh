#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe0=build/examples/test_uncertainty

images=("dat/pixabay/siggraph.png")
#"firefox-T.png")

# for high res
eps=0.001
rho=6.0
pcg_iters=5
low_thr=8

rm *BBFMM*.bin

# ---the larger low_thr, the sparser curves---
for input in "${images[@]}"; do
    filename=$(basename "${input}")
    name="${filename%.*}"
    echo "${name}"
    outdir=result/kl_res/uncertainty/${name}/
    mkdir -p ${outdir}
    ${exe0} outdir=${outdir} input=${input} low_threshold=${low_thr} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=10000000 lambda=0 pcg_iters=${pcg_iters} cg_iters=0 debug_fmm_eval=false test_cuda_extrap=false test_old_fmm=false 2>&1 | tee ${outdir}/log-thr-${low_thr}-rho-${rho}-pcg-${pcg_iters}.txt
done
