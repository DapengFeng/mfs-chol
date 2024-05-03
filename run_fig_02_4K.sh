#!/bin/bash

# 1M DoFs to solve: need a good GPU:)

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe0=build/examples/test_diffusion_curves

rm *BBFMM*.bin

images="dat/pixabay/moutain-ximengw-H.png"
filename=$(basename "${images}")
name="${filename%.*}"
echo "${name}"

eps=0.0001

for rho in 6.0; do
    for pcg_iters in 9; do
	for low_thr in 43; do
outdir=result/kl_res/large_example_laptop/${name}/thr-${low_thr}-rho-${rho}/
mkdir -p ${outdir}

${exe0} outdir=${outdir} input=${images} low_threshold=${low_thr} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=10000000 lambda=0 pcg_iters=${pcg_iters} cg_iters=20 debug_fmm_eval=false test_cuda_extrap=false test_old_fmm=false fmm_order=5 fmm_eps=1e-6 2>&1 | tee ${outdir}/log-thr-${low_thr}-rho-${rho}-pcg-${pcg_iters}.txt

	done
    done
done
