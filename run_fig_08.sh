#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

exe0=build/examples/test_diffusion_curves

images=("dat/pixabay/lotus-shanghaistoneman-M.jpg" "dat/pixabay/rose-pexels-M.jpg" "dat/pixabay/bananas-petelinforth-M.jpg")

# for high res
eps=0.0001
rho=6.0
pcg_iters=7

rm *BBFMM*.bin

# ---the larger low_thr, the sparser curves---
for input in "${images[@]}"; do
    filename=$(basename "${input}")
    name="${filename%.*}"
    echo "${name}"
for low_thr in 25; do
    outdir=result/kl_res/diffusion_points_newtiming/${name}/thr-${low_thr}
    mkdir -p ${outdir}
    ${exe0} outdir=${outdir} input=${input} low_threshold=${low_thr} num_sec=16 eps=${eps} rho=${rho} max_supernode_size=10000000 lambda=0 pcg_iters=${pcg_iters} cg_iters=0 debug_fmm_eval=false test_cuda_extrap=false test_old_fmm=false show_img=true 2>&1 | tee ${outdir}/log-thr-${low_thr}-rho-${rho}-pcg-${pcg_iters}.txt
done
done
