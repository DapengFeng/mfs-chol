#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)*0.6" | bc`
num_threads=$(round ${num_threads} 0)
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

outdir=result/kl_res/hpro3d-hlu/
mkdir -p ${outdir}

eps=0.0001

rm PBBFMM_laplace_3d*.bin

for guess_n in 050000 100000 150000 200000 250000 300000; do

echo "=============== ours ==================="
exe=build/examples/klchol_profiling
rho=2.5

pcg_iters=12
if [ ${guess_n} -ge 200000 ]; then
   pcg_iters=20
fi

${exe} outdir=${outdir} eps=${eps} guess_n=${guess_n} rho=${rho} pcg_iters=${pcg_iters} 2>&1 | tee ${outdir}/log-ours-${guess_n}-${rho}.txt
echo "========================================="

echo "=============== Hlibpro ================="
exe=build/examples/test_hpro3d

for acc_tol in 1e-2 1e-3 1e-4 1e-5; do
    for num_iters in 002 010 020 040 060 080 100; do
	${exe} outdir=${outdir} guess_n=${guess_n} eps=${eps} num_iters=${num_iters} acc_tol=${acc_tol} brute_force_eval=false 2>&1 | tee ${outdir}/log-hlu-${guess_n}-${acc_tol}-iters-${num_iters}.txt
    done
done
echo "========================================="

done
