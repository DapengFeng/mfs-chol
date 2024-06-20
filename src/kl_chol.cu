#include "kl_chol.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <Eigen/Sparse>
#include <thrust/reverse.h>
#include <thrust/binary_search.h>

#include "cublas_wrapper.h"

namespace klchol {

///-------------------- utility for local Cholesky ---------------------------
template <typename scalar_t, typename index_t>
struct dense_chol_fac;

template <typename index_t>
struct dense_chol_fac<double, index_t>
{
  __device__
  static void run(const index_t n, double * __restrict__ A)
  {
    // A = U^tU
    double dot_sum = 0;
    index_t j = 0, i = 0;
    for (j = 0; j < n; ++j) {
      // Uij
      for (i = 0; i < j; ++i) {
        dot_sum = thrust::inner_product(thrust::device, A+j*n, A+j*n+i, A+i*n, 0.0);
        A[i+j*n] = (A[i+n*j]-dot_sum)/A[i+i*n];
      }
      // Ujj
      dot_sum = thrust::inner_product(thrust::device, A+j*n, A+j*n+i, A+i*n, 0.0);
      A[i+j*n] = sqrt(A[j*n+j]-dot_sum);
    }
  }
};

template <typename index_t>
struct dense_chol_fac<thrust::complex<double>, index_t>
{
  typedef thrust::complex<double> Complex;

  __device__
  static void run(const index_t n, Complex * __restrict__ rA)
  {
    Complex dot_sum = 0;
    index_t j = 0, i = 0;
    for (j = 0; j < n; ++j) {
      // Uij
      for (i = 0; i < j; ++i) {
        dot_sum = thrust::inner_product(
            thrust::device, rA+i*n, rA+i*n+i, rA+j*n,
            Complex(0.0),
            thrust::plus<Complex>(),
            [](const Complex &a, const Complex &b)->Complex {
              return thrust::conj(a)*b;
            });
        rA[i+j*n] = (rA[i+n*j]-dot_sum)/rA[i+i*n];
      }
      // Ujj
      dot_sum = thrust::inner_product(
          thrust::device, rA+i*n, rA+i*n+i, rA+j*n,
          Complex(0.0),
          thrust::plus<Complex>(),
          [](const Complex &a, const Complex &b)->Complex {
            return thrust::conj(a)*b;
          });      
      rA[i+j*n] = sqrt((rA[j*n+j]-dot_sum).real());
    }
  }
};

template <typename scalar_t, typename index_t>
__device__ __host__
void upper_tri_solve(const index_t                nU,
                     const scalar_t* __restrict__ U,
                     const index_t                nb,
                     scalar_t*       __restrict__ b)
{
  // x = U^{-1}b
  const index_t end = nb-1;
  for (index_t i = 0; i < nb; ++i) {
    for (index_t j = 0; j < i; ++j) {
      b[end-i] -= b[end-j]*U[end-i+(end-j)*nU];
    }
    b[end-i] /= U[end-i+(end-i)*nU];
  }
}

// --------------------- cuda evaluation kernels ---------------------------

/* NOTE: observation and prediction points are separately stored */
template <enum PDE_TYPE pde>
__global__ void tiled_predict(
    const typename pde_trait<pde>::index_t                 n_pred,
    const typename pde_trait<pde>::real_t   * __restrict__ d_pred_xyz,
    const typename pde_trait<pde>::index_t                 n_train,
    const typename pde_trait<pde>::real_t   * __restrict__ d_train_xyz,
    const typename pde_trait<pde>::real_t   * __restrict__ d_train_nxyz,
    const typename pde_trait<pde>::real_t   * __restrict__ d_param,
    const int                                              impulse_dim,
    const typename pde_trait<pde>::scalar_t * __restrict__ d_x,
    typename pde_trait<pde>::scalar_t       * __restrict__ d_yKx,
    const int                                              kf_id)
{
  printf("[tiled_predict] Should never be here!\n");
  ASSERT(0);
}

template <>
__global__ void tiled_predict<PDE_TYPE::POISSON>(
    const pde_trait<PDE_TYPE::POISSON>::index_t                 n_pred,
    const pde_trait<PDE_TYPE::POISSON>::real_t   * __restrict__ d_pred_xyz,
    const pde_trait<PDE_TYPE::POISSON>::index_t                 n_train,
    const pde_trait<PDE_TYPE::POISSON>::real_t   * __restrict__ d_train_xyz,
    const pde_trait<PDE_TYPE::POISSON>::real_t   * __restrict__ d_train_nxyz,    
    const pde_trait<PDE_TYPE::POISSON>::real_t   * __restrict__ d_param,
    const int                                                   impulse_dim,
    const pde_trait<PDE_TYPE::POISSON>::scalar_t * __restrict__ d_x,
    pde_trait<PDE_TYPE::POISSON>::scalar_t       * __restrict__ d_yKx,
    const int                                                   kf_id)
{
  typedef pde_trait<PDE_TYPE::POISSON>::index_t  index_t;
  typedef pde_trait<PDE_TYPE::POISSON>::scalar_t scalar_t;
  typedef pde_trait<PDE_TYPE::POISSON>::real_t   real_t;
  
  const index_t pid = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real_t shared_xyzf[];
  
  Eigen::Matrix<real_t, 3, 1> xyz_p;
  xyz_p.setZero();
  if ( pid < n_pred ) {
    xyz_p[0] = d_pred_xyz[3*pid+0];
    xyz_p[1] = d_pred_xyz[3*pid+1];
    xyz_p[2] = d_pred_xyz[3*pid+2];
  }

  const int stride = 3+impulse_dim;  
  const real_t param[8] = {d_param[0], d_param[1], d_param[2], d_param[3],
                           d_param[4], d_param[5], d_param[6], d_param[7]};
  const index_t tile_num = (n_train+blockDim.x-1)/blockDim.x;

  if ( impulse_dim == 1 ) { // scalar impulse
    scalar_t dKx = 0;
    for (index_t tile = 0; tile < tile_num; ++tile) {
      const index_t thread_idx = threadIdx.x;
      const index_t pj = tile*blockDim.x+thread_idx;
      if ( pj < n_train ) {      
        shared_xyzf[stride*thread_idx+0] = d_train_xyz[3*pj+0];
        shared_xyzf[stride*thread_idx+1] = d_train_xyz[3*pj+1];
        shared_xyzf[stride*thread_idx+2] = d_train_xyz[3*pj+2];
        shared_xyzf[stride*thread_idx+3] = d_x[pj];
      } else {
        shared_xyzf[stride*thread_idx+0] = 0; 
        shared_xyzf[stride*thread_idx+1] = 0; 
        shared_xyzf[stride*thread_idx+2] = 0; 
        shared_xyzf[stride*thread_idx+3] = 0; 
      }    
      __syncthreads();

      scalar_t G;
      for (index_t k = 0; k < blockDim.x; ++k) {
        gf_summary<PDE_TYPE::POISSON, scalar_t, real_t>::run(
            kf_id, xyz_p.data(),
            &shared_xyzf[stride*k],
            nullptr, // TODO
            nullptr, // TODO
            &param[0], &G);
        dKx += G*shared_xyzf[stride*k+3];
      }
      __syncthreads();
    }

    if ( pid < n_pred ) {
      d_yKx[pid] = dKx;
    }

    return;
  }

  if ( impulse_dim == 3 ) { // vector impulse
    Eigen::Matrix<scalar_t, 3, 1> dKx;
    dKx.setZero();
    for (index_t tile = 0; tile < tile_num; ++tile) {
      const index_t thread_idx = threadIdx.x;
      const index_t pj = tile*blockDim.x+thread_idx;
      if ( pj < n_train ) {
        shared_xyzf[stride*thread_idx+0] = d_train_xyz[3*pj+0];
        shared_xyzf[stride*thread_idx+1] = d_train_xyz[3*pj+1];
        shared_xyzf[stride*thread_idx+2] = d_train_xyz[3*pj+2];
        shared_xyzf[stride*thread_idx+3] = d_x[3*pj+0];
        shared_xyzf[stride*thread_idx+4] = d_x[3*pj+1];
        shared_xyzf[stride*thread_idx+5] = d_x[3*pj+2];
      } else {
        shared_xyzf[stride*thread_idx+0] = 0; 
        shared_xyzf[stride*thread_idx+1] = 0; 
        shared_xyzf[stride*thread_idx+2] = 0; 
        shared_xyzf[stride*thread_idx+3] = 0; 
        shared_xyzf[stride*thread_idx+4] = 0; 
        shared_xyzf[stride*thread_idx+5] = 0; 
      }    
      __syncthreads();

      scalar_t G;
      for (index_t k = 0; k < blockDim.x; ++k) {
        gf_summary<PDE_TYPE::POISSON, scalar_t, real_t>::run(
            kf_id, xyz_p.data(), &shared_xyzf[stride*k],
            nullptr, nullptr, //TODO
            &param[0], &G);
        dKx += G*Eigen::Map<const Eigen::Matrix<scalar_t, 3, 1>>(&shared_xyzf[stride*k+3], 3);
      }
      __syncthreads();
    }

    if ( pid < n_pred ) {
      d_yKx[3*pid+0] = dKx.x();
      d_yKx[3*pid+1] = dKx.y();
      d_yKx[3*pid+2] = dKx.z();
    }

    return;
  }
}

template <>
__global__ void tiled_predict<PDE_TYPE::KELVIN>(
    const pde_trait<PDE_TYPE::KELVIN>::index_t                 n_pred,
    const pde_trait<PDE_TYPE::KELVIN>::real_t   * __restrict__ d_pred_xyz,
    const pde_trait<PDE_TYPE::KELVIN>::index_t                 n_train,
    const pde_trait<PDE_TYPE::KELVIN>::real_t   * __restrict__ d_train_xyz,
    const pde_trait<PDE_TYPE::KELVIN>::real_t   * __restrict__ d_train_nxyz,    
    const pde_trait<PDE_TYPE::KELVIN>::real_t   * __restrict__ d_param,
    const int                                                  impulse_dim,
    const pde_trait<PDE_TYPE::KELVIN>::scalar_t * __restrict__ d_x,
    pde_trait<PDE_TYPE::KELVIN>::scalar_t       * __restrict__ d_yKx,
    const int                                                  kf_id)
{
  typedef pde_trait<PDE_TYPE::KELVIN>::index_t  index_t;
  typedef pde_trait<PDE_TYPE::KELVIN>::scalar_t scalar_t;
  typedef pde_trait<PDE_TYPE::KELVIN>::real_t   real_t;
  
  const index_t pid = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real_t shared_xyzf[];
  
  Eigen::Matrix<real_t, 3, 1> xyz_p;
  xyz_p.setZero();
  if ( pid < n_pred ) {
    xyz_p[0] = d_pred_xyz[3*pid+0];
    xyz_p[1] = d_pred_xyz[3*pid+1];
    xyz_p[2] = d_pred_xyz[3*pid+2];
  }

  const int stride = 3+3*impulse_dim; // xyz and forces
  const real_t param[8] = {d_param[0], d_param[1], d_param[2], d_param[3],
                           d_param[4], d_param[5], d_param[6], d_param[7]};

  const index_t tile_num = (n_train+blockDim.x-1)/blockDim.x;

  if ( impulse_dim == 1 ) {
    Eigen::Matrix<scalar_t, 3, 1> dKx;
    dKx.setZero();
    for (index_t tile = 0; tile < tile_num; ++tile) {
      const index_t thread_idx = threadIdx.x;
      const index_t pj = tile*blockDim.x+thread_idx;
      if ( pj < n_train ) {
        shared_xyzf[stride*thread_idx+0] = d_train_xyz[3*pj+0];
        shared_xyzf[stride*thread_idx+1] = d_train_xyz[3*pj+1];
        shared_xyzf[stride*thread_idx+2] = d_train_xyz[3*pj+2];
        shared_xyzf[stride*thread_idx+3] = d_x[3*pj+0];
        shared_xyzf[stride*thread_idx+4] = d_x[3*pj+1];
        shared_xyzf[stride*thread_idx+5] = d_x[3*pj+2];
      } else {
        shared_xyzf[stride*thread_idx+0] = 0; 
        shared_xyzf[stride*thread_idx+1] = 0; 
        shared_xyzf[stride*thread_idx+2] = 0; 
        shared_xyzf[stride*thread_idx+3] = 0; 
        shared_xyzf[stride*thread_idx+4] = 0; 
        shared_xyzf[stride*thread_idx+5] = 0;
      }    
      __syncthreads();

      Eigen::Matrix<scalar_t, 3, 3> G;
      for (index_t k = 0; k < blockDim.x; ++k) {
        gf_summary<PDE_TYPE::KELVIN, scalar_t, real_t>::run(
            kf_id, xyz_p.data(),
            &shared_xyzf[stride*k],
            nullptr, nullptr,
            &param[0], G.data());
        dKx += G*Eigen::Map<const Eigen::Matrix<scalar_t, 3, 1>>(&shared_xyzf[stride*k+3], 3);
      }
      __syncthreads();
    }

    if ( pid < n_pred ) {
      d_yKx[3*pid+0] = dKx.x();
      d_yKx[3*pid+1] = dKx.y();
      d_yKx[3*pid+2] = dKx.z();
    }
  }
}

template <>
__global__ void tiled_predict<PDE_TYPE::HELMHOLTZ>(
    const pde_trait<PDE_TYPE::HELMHOLTZ>::index_t                 n_pred,
    const pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   * __restrict__ d_pred_xyz,
    const pde_trait<PDE_TYPE::HELMHOLTZ>::index_t                 n_train,
    const pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   * __restrict__ d_train_xyz,
    const pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   * __restrict__ d_train_nxyz,    
    const pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   * __restrict__ d_param,
    const int                                                     impulse_dim,
    const pde_trait<PDE_TYPE::HELMHOLTZ>::scalar_t * __restrict__ d_x,
    pde_trait<PDE_TYPE::HELMHOLTZ>::scalar_t       * __restrict__ d_yKx,
    const int                                                     kf_id)
{
  typedef pde_trait<PDE_TYPE::HELMHOLTZ>::index_t  index_t;
  typedef pde_trait<PDE_TYPE::HELMHOLTZ>::scalar_t scalar_t;
  typedef pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   real_t;

  const index_t pid = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real_t shared_xyzf[];
  
  Eigen::Matrix<real_t, 3, 1> xyz_p;
  xyz_p.setZero();
  if ( pid < n_pred ) {
    xyz_p[0] = d_pred_xyz[3*pid+0];
    xyz_p[1] = d_pred_xyz[3*pid+1];
    xyz_p[2] = d_pred_xyz[3*pid+2];
  }

  const int stride = 3+3+impulse_dim*2;  
  const real_t param[8] = {
    d_param[0], d_param[1], d_param[2], d_param[3],
    d_param[4], d_param[5], 0.0,        1.0
  };
  const index_t tile_num = (n_train+blockDim.x-1)/blockDim.x;

  if ( impulse_dim == 1 ) { // scalar impulse
    scalar_t dKx = 0;
    for (index_t tile = 0; tile < tile_num; ++tile) {
      const index_t thread_idx = threadIdx.x;
      const index_t pj = tile*blockDim.x+thread_idx;
      if ( pj < n_train ) {      
        shared_xyzf[stride*thread_idx+0] = d_train_xyz[3*pj+0];
        shared_xyzf[stride*thread_idx+1] = d_train_xyz[3*pj+1];
        shared_xyzf[stride*thread_idx+2] = d_train_xyz[3*pj+2];
        shared_xyzf[stride*thread_idx+3] = d_train_nxyz[3*pj+0];
        shared_xyzf[stride*thread_idx+4] = d_train_nxyz[3*pj+1];
        shared_xyzf[stride*thread_idx+5] = d_train_nxyz[3*pj+2];        
        shared_xyzf[stride*thread_idx+6] = d_x[pj].real();
        shared_xyzf[stride*thread_idx+7] = d_x[pj].imag();
      } else {
        shared_xyzf[stride*thread_idx+0] = 0;
        shared_xyzf[stride*thread_idx+1] = 0;
        shared_xyzf[stride*thread_idx+2] = 0;
        shared_xyzf[stride*thread_idx+3] = 0;
        shared_xyzf[stride*thread_idx+4] = 0; 
        shared_xyzf[stride*thread_idx+5] = 0;
        shared_xyzf[stride*thread_idx+6] = 0;
        shared_xyzf[stride*thread_idx+7] = 0;       
      }    
      __syncthreads();

      scalar_t G;
      real_t r_xy[3];
      for (index_t k = 0; k < blockDim.x; ++k) {
        r_xy[0] = xyz_p[0]-shared_xyzf[stride*k+0];
        r_xy[1] = xyz_p[1]-shared_xyzf[stride*k+1];
        r_xy[2] = xyz_p[2]-shared_xyzf[stride*k+2];
        gf_summary<PDE_TYPE::HELMHOLTZ, scalar_t, real_t>::run
            (kf_id, xyz_p.data(), &shared_xyzf[stride*k+0],
             r_xy, &shared_xyzf[stride*k+3],
             &param[0], &G);
        dKx += G*scalar_t(shared_xyzf[stride*k+6], shared_xyzf[stride*k+7]);
      }
      __syncthreads();
    }

    if ( pid < n_pred ) {
      d_yKx[pid] = dKx;
    }

    return;
  }
}

template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void sectioned_eval_Kx(const index_t        n_point,
                                  const index_t        n_section,
                                  const real_t   * __restrict__ d_xyz,
                                  const real_t   * __restrict__ d_nxyz,
                                  const real_t   * __restrict__ d_param,
                                  const scalar_t * __restrict__ d_x,
                                  scalar_t       * __restrict__ d_Kx,
                                  const int                     kf_id)

{
  const index_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( tid >= n_point*n_section ) {
    return;
  }

  const index_t sec_size = (n_point+n_section-1)/n_section;
  
  const index_t pid = tid%n_point;
  const index_t sec_id = tid/n_point;
  const index_t sec_begin = sec_id*sec_size;
  const index_t sec_end = min((sec_id+1)*sec_size, n_point);

  const int d = pde_trait<pde>::d;
  typedef Eigen::Matrix<scalar_t, d, d> mat_t;
  typedef Eigen::Matrix<scalar_t, d, 1> vec_t;

  vec_t res;
  res.setZero();

  mat_t G;  
  for (index_t j = sec_begin; j < sec_end; ++j) {
    gf_summary<pde, scalar_t, real_t>::run(
        kf_id, &d_xyz[3*pid], &d_xyz[3*j], &d_nxyz[3*pid], &d_nxyz[3*j],
        d_param, G.data());
    res += G*Eigen::Map<const vec_t>(&d_x[d*j], d);
  }

  #pragma unroll (pde_trait<pde>::d)
  for (index_t k = 0; k < d; ++k) {
    d_Kx[d*pid+d*n_point*sec_id+k] = *(res.data()+k);
  }
}

template <typename scalar_t, typename index_t>
static void sectioned_eval_reduction(const cublasHandle_t &handle,
                                     const index_t   n,
                                     const index_t   n_section,
                                     const scalar_t * __restrict__ d_sec_eval,
                                     scalar_t       * __restrict__ d_res)
{
  const scalar_t ALPHA = 1;
  CHECK_CUDA(cudaMemset(d_res, 0, n*sizeof(scalar_t)));
  for (index_t i = 0; i < n_section; ++i) {
    CHECK_CUBLAS(cublasAxpyEx(handle, n, &ALPHA, cuda_data_trait<scalar_t>::dataType,
                              d_sec_eval+n*i, cuda_data_trait<scalar_t>::dataType, 1,
                              d_res, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
  }
}

///-------------------- assemble least-square covariance ------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void ker_least_square_lhs(const index_t               n_src,
                                     const real_t * __restrict__ d_src_xyz,
                                     const real_t * __restrict__ d_src_nxyz,
                                     const index_t               n_bnd,
                                     const real_t * __restrict__ d_bnd_xyz,
                                     const real_t * __restrict__ d_bnd_nxyz,
                                     const real_t * __restrict__ d_param,
                                     scalar_t     * __restrict__ d_K_val,
                                     const int kf_id)
{
  const index_t iter = blockIdx.x*blockDim.x + threadIdx.x;
  const index_t nnz = n_bnd*n_src;
  if ( iter >= nnz ) {
    return;
  }
  
  const index_t col = iter/n_bnd, row = iter%n_bnd;
  const int d = pde_trait<pde>::d;
  const index_t stride = d*n_bnd;
  
  Eigen::Matrix<scalar_t, d, d> G;
  gf_summary<pde, scalar_t, real_t>::run(
      kf_id,
      &d_bnd_xyz[3*row],  &d_src_xyz[3*col],
      &d_bnd_nxyz[3*row], &d_src_nxyz[3*col],
      d_param, G.data());

  #pragma unroll (pde_trait<pde>::d*pde_trait<pde>::d)
  for (index_t i = 0; i < d*d; ++i) {
    const index_t c = i/d, r = i%d;
    d_K_val[stride*(d*col+c)+d*row+r] = *(G.data()+i);
  }  
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::build_cov_mat_LS(const KERNEL_TYPE ker_type,
                                          const real_t *h_aux,
                                          const index_t length)
{
  ASSERT(d_bnd_xyz_ && d_src_xyz_); // boundary and source points must be present for LS solve

  const int d = pde_trait<pde>::d;
  const index_t nnz = n_bnd_pts_*n_src_pts_;
  const real_t Gb = 1.0*(d*d)*nnz*sizeof(scalar_t)/1000/1000/1000;
  std::cout << "# number of source points=" << n_src_pts_ << std::endl;
  std::cout << "# mem for kernel mat=" << Gb << " Gb" << std::endl;

  // rows and cols of the constraint matrix
  K_rows_ = d*n_bnd_pts_;
  K_cols_ = d*n_src_pts_;
          
  if ( d_K_val_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_K_val_, K_rows_*K_cols_*sizeof(scalar_t)));
    CHECK_CUDA(cudaMemset(d_K_val_, 0, K_rows_*K_cols_*sizeof(scalar_t)));
  }

  ASSERT(length <= AUX_SIZE_);  
  if ( d_aux_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_aux_, AUX_SIZE_*sizeof(real_t)));
  }
  CHECK_CUDA(cudaMemcpy(d_aux_, h_aux, length*sizeof(real_t), cudaMemcpyHostToDevice));
  
  const index_t threads_num = 256;
  const index_t blocks_num = (nnz+threads_num-1)/threads_num;
  ker_least_square_lhs<pde, scalar_t, index_t, real_t>
      <<< blocks_num, threads_num >>>
      (n_src_pts_, d_src_xyz_, d_src_nxyz_,
       n_bnd_pts_, d_bnd_xyz_, d_bnd_nxyz_,
       d_aux_, d_K_val_, static_cast<index_t>(ker_type));
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::K(scalar_t *Kval) const
{
  CHECK_CUDA(cudaMemcpy(Kval, d_K_val_, K_rows_*K_cols_*sizeof(scalar_t), cudaMemcpyDeviceToHost));  
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::build_cov_rhs_LS(const scalar_t *h_rhs_bnd,
                                          const index_t   size_bnd,
                                          scalar_t       *h_rhs_src,
                                          const index_t   size_src)
{  
  const int d = pde_trait<pde>::d;
  ASSERT(size_bnd == d*n_bnd_pts_ && size_src == d*n_src_pts_);

  if ( d_rhs_bnd_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_rhs_bnd_, size_bnd*sizeof(scalar_t)));
  }
  if ( d_rhs_src_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_rhs_src_, size_src*sizeof(scalar_t)));
  }

  // copy rhs from boundary and evaluate  
  const scalar_t ALPHA = 1, BETA = 0; 
  CHECK_CUDA(cudaMemcpy(d_rhs_bnd_, h_rhs_bnd, size_bnd*sizeof(scalar_t), cudaMemcpyHostToDevice));
  GEMV<scalar_t>::run(
      handle_,
      cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
      K_rows_, K_cols_,
      &ALPHA,
      d_K_val_, K_rows_,
      d_rhs_bnd_, 1,
      &BETA,
      d_rhs_src_, 1);
  CHECK_CUDA(cudaMemcpy(h_rhs_src, d_rhs_src_, size_src*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::debug(const KERNEL_TYPE ker_type) const
{
#if 0
  using namespace std;
  using namespace Eigen;

  const int dim_ = pde_trait<pde>::d;
  
  const size_t nnz = (1+n_src_pts_)*n_src_pts_/2;
  VectorXi ptr(n_src_pts_+1);
  CHECK_CUDA(cudaMemcpy(ptr.data(), d_K_ptr_, ptr.size()*sizeof(int), cudaMemcpyDeviceToHost));
  //  std::cout << ptr.transpose() << std::endl << std::endl;
  
  VectorXd val(dim_*dim_*nnz);
  CHECK_CUDA(cudaMemcpy(val.data(), d_K_val_, val.size()*sizeof(double), cudaMemcpyDeviceToHost));
  //  std::cout << val.head(60).transpose() << std::endl << std::endl;
  
  VectorXd xyz(3*n_src_pts_);
  CHECK_CUDA(cudaMemcpy(xyz.data(), d_src_xyz_, xyz.size()*sizeof(double), cudaMemcpyDeviceToHost));
  //  std::cout << xyz.head(60).transpose() << std::endl << std::endl;
  
  VectorXd param(AUX_SIZE_);
  CHECK_CUDA(cudaMemcpy(param.data(), d_aux_, param.size()*sizeof(double), cudaMemcpyDeviceToHost));
  //  std::cout << param.transpose() << std::endl << std::endl;

  const int d = pde_trait<pde>::d;
  typedef Eigen::Matrix<double, d, d> G_type;
  
  for (size_t j = 0; j < n_src_pts_; ++j) {
    for (size_t i = 0; i <= j; ++i) {
      G_type G_store = Eigen::Map<G_type>(&val[d*d*(ptr[j]+i)], d, d),
          G = G_type::Zero();
      gf_summary<pde, scalar_t, real_t>::run(ker_type, &xyz[3*j], &xyz[3*i],
                                             param.data(), G.data());
      if ( (G-G_store).norm() > 1e-6 ) {
        cerr << "# weird unmatch" << j << ", " << i << endl;
        return;
      }
    }
  }
  std::cout << "debug success" << std::endl;
#endif
}

///------------------- supernodal ------------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void klchol_super_fac_asm_scalar(const index_t  n_super,
                                            const index_t * __restrict__ d_super_ptr,
                                            const index_t * __restrict__ d_super_ind,
                                            const index_t * __restrict__ d_ptr,
                                            const index_t * __restrict__ d_ind,
                                            scalar_t      * __restrict__ d_val,
                                            const real_t  * __restrict__ d_xyz,
                                            const real_t  * __restrict__ d_nxyz,
                                            const real_t  * __restrict__ d_param,
                                            const index_t * __restrict__ d_TH_ptr,
                                            scalar_t      * __restrict__ d_TH_val,
                                            const int kf_id,
                                            const index_t                ROWS_COV,
                                            const scalar_t *__restrict__ d_LS_COV)
{
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }

  const auto TH_iter = d_TH_ptr[sid];
  const auto first_dof = d_super_ind[d_super_ptr[sid]];
  const index_t local_n = d_ptr[first_dof+1]-d_ptr[first_dof];

  if ( d_LS_COV == NULL ) {
    for (auto iter_i = d_ptr[first_dof], p = 0; iter_i < d_ptr[first_dof+1]; ++iter_i, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[first_dof+1]; ++iter_j, ++q) {
        const auto I = d_ind[iter_i], J = d_ind[iter_j];
        scalar_t G = 0;
        gf_summary<pde, scalar_t, real_t>::run(
            kf_id, &d_xyz[3*I], &d_xyz[3*J], &d_nxyz[3*I], &d_nxyz[3*J],
            d_param, &G);

        // reverse ordering for THETA
        const index_t rev_p = local_n-1-p, rev_q = local_n-1-q;
        d_TH_val[TH_iter+rev_p+rev_q*local_n] = G;
        conjugation<scalar_t>()(G);
        d_TH_val[TH_iter+rev_q+rev_p*local_n] = G;
      }
    }
  } else {
    for (auto iter_i = d_ptr[first_dof], p = 0; iter_i < d_ptr[first_dof+1]; ++iter_i, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[first_dof+1]; ++iter_j, ++q) {
        const auto I = d_ind[iter_i], J = d_ind[iter_j];
        scalar_t G = 0;
        gf_summary<pde, scalar_t, real_t>::run(I, J, ROWS_COV/pde_trait<pde>::d, d_LS_COV, &G);

        // reverse ordering for THETA
        const index_t rev_p = local_n-1-p, rev_q = local_n-1-q;
        d_TH_val[TH_iter+rev_p+rev_q*local_n] = G;
        conjugation<scalar_t>()(G);
        d_TH_val[TH_iter+rev_q+rev_p*local_n] = G;
      }
    }
  }
}

template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void klchol_super_fac_asm_vector(const index_t  n_super,
                                            const index_t * __restrict__ d_super_ptr,
                                            const index_t * __restrict__ d_super_ind,
                                            const index_t * __restrict__ d_ptr,
                                            const index_t * __restrict__ d_ind,
                                            scalar_t      * __restrict__ d_val,
                                            const real_t  * __restrict__ d_xyz,
                                            const real_t  * __restrict__ d_nxyz,
                                            const real_t  * __restrict__ d_param,
                                            const index_t * __restrict__ d_TH_ptr,
                                            scalar_t      * __restrict__ d_TH_val,
                                            const int kf_id,
                                            const index_t                ROWS_COV,
                                            const scalar_t *__restrict__ d_LS_COV)
{
  typedef Eigen::Matrix<scalar_t, 3, 3> Mat3f;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vec3f;  
  
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }

  const auto TH_iter = d_TH_ptr[sid];
  const auto first_dof = d_super_ind[d_super_ptr[sid]];
  const index_t local_n = d_ptr[3*first_dof+1]-d_ptr[3*first_dof];

  if ( d_LS_COV == NULL ) {
    for (auto iter_i = d_ptr[3*first_dof], p = 0; iter_i < d_ptr[3*first_dof+1]; iter_i += 3, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[3*first_dof+1]; iter_j += 3, ++q) {
        const auto I = d_ind[iter_i]/3, J = d_ind[iter_j]/3;

        Mat3f G = Mat3f::Zero();
        gf_summary<pde, scalar_t, real_t>::run(
            kf_id, &d_xyz[3*I], &d_xyz[3*J], &d_nxyz[3*I], &d_nxyz[3*J],
            d_param, G.data());

        // reverse ordering for THETA
        index_t rev_p, rev_q;

        rev_p = local_n/3-1-p; rev_q = local_n/3-1-q;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(0, 1);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(1, 0);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(0, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(2, 0);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(1, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(2, 1);
        }

        rev_p = local_n/3-1-q; rev_q = local_n/3-1-p;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(1, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(0, 1);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(2, 0);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(0, 2);
          
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(2, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(1, 2);
        }
      }
    }
  } else {
    for (auto iter_i = d_ptr[3*first_dof], p = 0; iter_i < d_ptr[3*first_dof+1]; iter_i += 3, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[3*first_dof+1]; iter_j += 3, ++q) {
        const auto I = d_ind[iter_i]/3, J = d_ind[iter_j]/3;

        Mat3f G = Mat3f::Zero();
        gf_summary<pde, scalar_t, real_t>::run(I, J, ROWS_COV/pde_trait<pde>::d, d_LS_COV, G.data());
        
        // reverse ordering for THETA, G could be unsymmetric!!!
        index_t rev_p, rev_q;
        
        rev_p = local_n/3-1-p; rev_q = local_n/3-1-q;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(0, 1);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(1, 0);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(0, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(2, 0);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(1, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(2, 1);
        }

        rev_p = local_n/3-1-q; rev_q = local_n/3-1-p;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(1, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(0, 1);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(2, 0);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(0, 2);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(2, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(1, 2);
        }
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void klchol_super_fac_chol(const index_t n_super,
                                      const index_t * __restrict__ d_TH_ptr,
                                      scalar_t      * __restrict__ d_TH_val)
{
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }
  const index_t local_n = floor(sqrt((double)(d_TH_ptr[sid+1]-d_TH_ptr[sid]))+0.5);
  const auto TH_iter = d_TH_ptr[sid];
  dense_chol_fac<scalar_t, index_t>::run(local_n, &d_TH_val[TH_iter]);
}

template <typename scalar_t, typename index_t>
__global__ void klchol_super_fac_bs(const index_t n,
                                    const index_t dim,
                                    const index_t * __restrict__  d_ptr,
                                    const index_t * __restrict__  d_ind,
                                    scalar_t      * __restrict__  d_val,
                                    const index_t * __restrict__  d_super_parent,
                                    const index_t * __restrict__  d_TH_ptr,
                                    const scalar_t * __restrict__ d_TH_val)
{
  const index_t pid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( pid >= n ) {
    return;
  }

  const index_t node_id = pid/dim;
  const index_t sid = d_super_parent[node_id];
  const index_t nU = floor(sqrt((double)(d_TH_ptr[sid+1]-d_TH_ptr[sid]))+0.5);
  const index_t nb = d_ptr[pid+1]-d_ptr[pid]; 
  const auto ptr_pid = d_ptr[pid];

  d_val[d_ptr[pid+1]-1] = 1.0;
  upper_tri_solve(nU, &d_TH_val[d_TH_ptr[sid]], nb, &d_val[ptr_pid]);
  thrust::reverse(thrust::device, d_val+ptr_pid, d_val+ptr_pid+nb);
  thrust::for_each(thrust::device, d_val+ptr_pid, d_val+ptr_pid+nb, conjugation<scalar_t>());
}

//==============================================================
template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::gpu_simpl_klchol(const index_t npts, const index_t ker_dim, const index_t num_sec)
    : npts_(npts), ker_dim_(ker_dim), n_(npts*ker_dim), num_sec_(num_sec)
{
  ASSERT(num_sec_ <= npts_);
  
  // query devices
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  }
  spdlog::info("GPU KL-Cholesky, npts={0:d}, ker_dim={1:d}, n={2:d}", npts_, ker_dim_, n_);

  // init cusparse and cusolverdn handles
  CHECK_CUSPARSE(cusparseCreate(&cus_handle_));
  CHECK_CUBLAS(cublasCreate(&bls_handle_));
  
  // malloc b and x: Nx1 buffer
  CHECK_CUDA(cudaMalloc((void **)&d_vecB_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_vecX_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemset(d_vecB_, 0, n_*sizeof(scalar_t)));  
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_*sizeof(scalar_t)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&b_, n_, d_vecB_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&x_, n_, d_vecX_, cuda_data_trait<scalar_t>::dataType));

  //  // triangular solve
  //  CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv_desc_));

  // malloc buffer for *partitioned* evaluation
  CHECK_CUDA(cudaMalloc((void **)&d_sec_eval_, n_*num_sec_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemset(d_sec_eval_, 0, n_*num_sec_*sizeof(scalar_t)));

  // vectors for pcg
  CHECK_CUDA(cudaMalloc((void **)&d_resd_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_temp_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_z_,    n_*sizeof(scalar_t)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&resd_, n_, d_resd_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&temp_, n_, d_temp_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&z_, n_, d_z_, cuda_data_trait<scalar_t>::dataType));

  // explicit covariance matrix
  ls_cov_ = new covariance_t();
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::~gpu_simpl_klchol()
{
  if ( cus_handle_ ) { CHECK_CUSPARSE(cusparseDestroy(cus_handle_)); }
  if ( bls_handle_ ) { CHECK_CUBLAS(cublasDestroy(bls_handle_)); }
  
  if ( d_ptr_) { CHECK_CUDA(cudaFree(d_ptr_)); }
  if ( d_ind_) { CHECK_CUDA(cudaFree(d_ind_)); }
  if ( d_val_) { CHECK_CUDA(cudaFree(d_val_)); }
  
  if ( d_vecB_) { CHECK_CUDA(cudaFree(d_vecB_)); }
  if ( d_vecX_) { CHECK_CUDA(cudaFree(d_vecX_)); }

  if ( d_resd_) { CHECK_CUDA(cudaFree(d_resd_)); }
  if ( d_temp_) { CHECK_CUDA(cudaFree(d_temp_)); }
  if ( d_z_   ) { CHECK_CUDA(cudaFree(d_z_));    }

  if ( d_ker_aux_ ) { CHECK_CUDA(cudaFree(d_ker_aux_)); }
  
  if ( d_xyz_ )  { CHECK_CUDA(cudaFree(d_xyz_));  }
  if ( d_nxyz_ ) { CHECK_CUDA(cudaFree(d_nxyz_)); }

  if ( d_pred_xyz_ ) { CHECK_CUDA(cudaFree(d_pred_xyz_)); }
  if ( d_pred_f_   ) { CHECK_CUDA(cudaFree(d_pred_f_));   }

  if ( d_TH_ptr_ ) { CHECK_CUDA(cudaFree(d_TH_ptr_)); }
  if ( d_TH_val_ ) { CHECK_CUDA(cudaFree(d_TH_val_)); }

  if ( d_work_) { CHECK_CUDA(cudaFree(d_work_)); }
  //  if ( d_sv_work_ ) { CHECK_CUDA(cudaFree(d_sv_work_)); }
  
  if ( A_ ) { CHECK_CUSPARSE(cusparseDestroySpMat(A_)); }
  if ( b_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(b_)); }
  if ( x_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(x_)); }

  //  if ( spsv_desc_ ) { CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsv_desc_)); }

  if ( resd_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(resd_)); }
  if ( temp_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(temp_)); }
  if ( z_    ) { CHECK_CUSPARSE(cusparseDestroyDnVec(z_));    }

  if ( ls_cov_ ) { delete ls_cov_; }
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_kernel(const KERNEL_TYPE ker_type,
                                       const real_t      *h_ker_aux,
                                       const index_t     length)
{
  ASSERT(length <= AUX_LENGTH_);
  ASSERT((ker_dim_ == 3) == (ker_type == KERNEL_TYPE::KELVIN_3D));

  if ( d_ker_aux_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_ker_aux_, AUX_LENGTH_*sizeof(real_t)));
  }
  ker_type_ = ker_type;
  CHECK_CUDA(cudaMemcpy(d_ker_aux_, h_ker_aux, length*sizeof(real_t), cudaMemcpyHostToDevice));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_source_points(const index_t npts, const real_t *h_xyz, const real_t *h_nxyz)
{
  ASSERT(npts_ == npts);

  if ( d_xyz_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_xyz_, 3*npts*sizeof(real_t)));
  }
  CHECK_CUDA(cudaMemcpy(d_xyz_, h_xyz, 3*npts*sizeof(real_t), cudaMemcpyHostToDevice));

  // fmm part
  sources_.resize(npts);
  srcs_.resize(npts);  
  #pragma omp parallel for
  for (index_t i = 0; i < npts; ++i) {
    sources_[i].x = h_xyz[3*i+0];
    sources_[i].y = h_xyz[3*i+1];
    sources_[i].z = h_xyz[3*i+2];

    srcs_[i].x = h_xyz[3*i+0];
    srcs_[i].y = h_xyz[3*i+1];
    srcs_[i].z = h_xyz[3*i+2];
  }

  // normals
  if ( d_nxyz_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_nxyz_, 3*npts*sizeof(real_t)));
  }
  CHECK_CUDA(cudaMemset(d_nxyz_, 0, 3*npts*sizeof(real_t)));  
  if ( h_nxyz ) { // if normal is present
    CHECK_CUDA(cudaMemcpy(d_nxyz_, h_nxyz, 3*npts*sizeof(real_t), cudaMemcpyHostToDevice));    
  }
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_target_points(const index_t npts, const real_t *h_xyz)
{
  if ( npts_pred_ == 0 ) { // for the first time
    npts_pred_ = npts;
  } else {
    ASSERT(npts_pred_ == npts);
  }

  const int d = pde_trait<pde>::d;
  if ( d_pred_xyz_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_pred_xyz_, 3*npts*sizeof(real_t)));
    CHECK_CUDA(cudaMalloc((void **)&d_pred_f_,   d*npts*sizeof(scalar_t)));
  }
  CHECK_CUDA(cudaMemcpy(d_pred_xyz_, h_xyz, 3*npts*sizeof(real_t), cudaMemcpyHostToDevice));

  // fmm parts
  targets_.resize(npts);
  tgts_.resize(npts);
  #pragma omp parallel for
  for (index_t i = 0; i < npts; ++i) {
    targets_[i].x = h_xyz[3*i+0];
    targets_[i].y = h_xyz[3*i+1];
    targets_[i].z = h_xyz[3*i+2];

    tgts_[i].x = h_xyz[3*i+0];
    tgts_[i].y = h_xyz[3*i+1];
    tgts_[i].z = h_xyz[3*i+2];
  }  
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_sppatt(const index_t n,
                                  const index_t nnz,
                                  const index_t *h_ptr,
                                  const index_t *h_ind)
{
  ASSERT(n == n_);
  spdlog::info("[set_sppatt] n={}, nnz={}", n, nnz);

  const auto &malloc_factor =
      [&]() {
        CHECK_CUDA(cudaMalloc((void **)&d_ptr_, (n_+1)*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_ind_, nnz_*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_val_, nnz_*sizeof(scalar_t)));
        CHECK_CUSPARSE(cusparseCreateCsr(&A_, n_, n_, nnz_, d_ptr_, d_ind_, d_val_,
                                         cuda_index_trait<index_t>::indexType,
                                         cuda_index_trait<index_t>::indexType,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_data_trait<scalar_t>::dataType));

        // cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
        // CHECK_CUSPARSE(cusparseSpMatSetAttribute(A_, CUSPARSE_SPMAT_FILL_MODE,
        //                                          &fillmode, sizeof(fillmode)));
        // cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
        // CHECK_CUSPARSE(cusparseSpMatSetAttribute(A_, CUSPARSE_SPMAT_DIAG_TYPE,
        //                                          &diagtype, sizeof(diagtype)));
      };
  
  const auto &malloc_theta = 
      [&]() {
        // for each spatial point...
        std::vector<index_t> ptr(npts_+1, 0);
        for (index_t j = 0; j < npts_; ++j) {
          const index_t nnz_j = h_ptr[ker_dim_*j+1]-h_ptr[ker_dim_*j];
          ASSERT(nnz_j%ker_dim_ == 0);
          ptr[j+1] = ptr[j]+nnz_j*nnz_j;
        }
        TH_nnz_ = ptr.back();
        const real_t Gb = TH_nnz_*sizeof(scalar_t)/(1024*1024*1024);
        spdlog::info("total size for THETA={0:.1f}, {1:.2f}", TH_nnz_, Gb);
        ASSERT(TH_nnz_ < INT_MAX);
        
        CHECK_CUDA(cudaMalloc((void **)&d_TH_ptr_, ptr.size()*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_TH_val_, ptr.back()*sizeof(scalar_t)));
        CHECK_CUDA(cudaMemcpy(d_TH_ptr_, &ptr[0], ptr.size()*sizeof(index_t), cudaMemcpyHostToDevice));
      };

  const auto &malloc_mv_buffer =
      [&]() {
        const scalar_t alpha = 1, beta = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            cus_handle_,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A_, b_, &beta, x_,
            cuda_data_trait<scalar_t>::dataType,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &buff_sz_));    
        CHECK_CUDA(cudaMalloc(&d_work_, buff_sz_));
        spdlog::info("SpMV buffer size={}", buff_sz_);
      };

  const auto &malloc_sv_buffer =
      [&]() {
        // const scalar_t alpha = 1;
        // CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        //     cus_handle_,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     A_, b_, x_,
        //     cuda_data_trait<scalar_t>::dataType,
        //     CUSPARSE_SPSV_ALG_DEFAULT,
        //     spsv_desc_,
        //     &sv_buff_sz_));
        // CHECK_CUDA(cudaMalloc(&d_sv_work_, sv_buff_sz_));
        // spdlog::info("SpSV buffer size={}", sv_buff_sz_);
      };

  if ( d_ptr_ == NULL ) { // allocate for the first time
    nnz_ = nnz;

    malloc_factor();
    malloc_theta();
    malloc_mv_buffer();
    malloc_sv_buffer();
  }

  if ( d_ptr_ && nnz_ != nnz ) { // reallocate if nnz is changed
    nnz_ = nnz;

    CHECK_CUDA(cudaFree(d_ptr_));
    CHECK_CUDA(cudaFree(d_ind_));
    CHECK_CUDA(cudaFree(d_val_));
    CHECK_CUSPARSE(cusparseDestroySpMat(A_));
    malloc_factor();

    CHECK_CUDA(cudaFree(d_TH_ptr_));
    CHECK_CUDA(cudaFree(d_TH_val_));
    malloc_theta();

    CHECK_CUDA(cudaFree(d_work_));
    malloc_mv_buffer();

    // CHECK_CUDA(cudaFree(d_sv_work_));
    malloc_sv_buffer();
  }

  // copy factor sparsity
  CHECK_CUDA(cudaMemcpy(d_ptr_, h_ptr, (n_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_ind_, h_ind, nnz_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_val_, 0, nnz_*sizeof(scalar_t)));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::get_factor(Eigen::SparseMatrix<scalar_t> &L) const
{
  Eigen::Matrix<index_t, -1, 1> ptr(n_+1), ind(nnz_);
  Eigen::Matrix<scalar_t, -1, 1> val(nnz_);  

  CHECK_CUDA(cudaMemcpy(ptr.data(), d_ptr_, (n_+1)*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ind.data(), d_ind_, nnz_*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(val.data(), d_val_, nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  L = Eigen::Map<Eigen::SparseMatrix<scalar_t>>(n_, n_, nnz_, ptr.data(), ind.data(), val.data());
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::real_t gpu_simpl_klchol<pde>::memory() const {
  return 0;
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::debug(const scalar_t *aux, const int n)
{
#if 0
  Eigen::VectorXi th_ptr(npts_+1);
  CHECK_CUDA(cudaMemcpy(th_ptr.data(), d_TH_ptr_, th_ptr.size()*sizeof(index_t), cudaMemcpyDeviceToHost));
  Eigen::VectorXd th_val(th_ptr[npts_]);
  std::cout << "thval size=" << th_val.size() << std::endl;
  CHECK_CUDA(cudaMemcpy(th_val.data(), d_TH_val_, th_val.size()*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  Eigen::VectorXi ptr(n_+1), ind(nnz_);
  Eigen::VectorXd val(nnz_);
  CHECK_CUDA(cudaMemcpy(ptr.data(), d_ptr_, (n_+1)*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ind.data(), d_ind_, nnz_*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(val.data(), d_val_, nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost));
  std::cout << "VAL norm=" << val.norm() << std::endl;

  for (size_t j = 0; j < npts_; ++j) {
    index_t start = th_ptr[j];
    index_t end = th_ptr[j+1];
    const size_t local_size = end-start, local_n = sqrt(local_size);
    std::cout << "local n=" << local_n << std::endl;
    std::cout << th_val.segment(start, local_size).reshaped(local_n, local_n) << std::endl << std::endl;

    size_t n = ptr[j+1]-ptr[j];
    ASSERT(n == local_n);
    Eigen::MatrixXd A(n, n);
    for (size_t iter1 = ptr[j], p = 0; iter1 < ptr[j+1]; ++iter1, ++p) {
      for (size_t iter2 = ptr[j], q = 0; iter2 < ptr[j+1]; ++iter2, ++q) {
        int I = ind[iter1], J = ind[iter2];
        reg_laplace_sqr(&x[3*I], &x[3*J], param, &A(p, q));
      }
    }
    std::cout << Eigen::MatrixXd(A.llt().matrixL()).transpose() << std::endl;

    Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
    e[0] = 1;
    Eigen::VectorXd tmp = A.llt().solve(e);
    std::cout << tmp.transpose()/sqrt(e.dot(tmp)) << std::endl << std::endl;;

    std::cout << val.segment(ptr[j], n).transpose() << std::endl;
    getchar();
  }
#endif
  //  CHECK_CUDA(cudaMemcpy(d_val_, h_valA, nnz_*sizeof(scalar_t), cudaMemcpyHostToDevice));
  //  CHECK_CUDA(cudaMemcpy(d_vecB_, h_valB, n_*sizeof(scalar_t), cudaMemcpyHostToDevice));

  // scalar_t alpha = 1.0, beta = 0;
  // CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                             &alpha, A_, b_, &beta, x_, CUDA_R_32F,
  //                             CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
        
  // scalar_t norm = 0;
  // CHECK_CUBLAS(cublasSnrm2(cub_handle_, n_, d_vecX_, 1, &norm));
  // std::cout << norm << std::endl;

  // int bufferSize = 0;
  // int *info = NULL;
  // scalar_t *buffer = NULL;
  // cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  // CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cud_handle_, uplo, n_, d_val_, n_, &bufferSize));
  // CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
  // CHECK_CUDA(cudaMalloc(&buffer, sizeof(scalar_t)*bufferSize));
  // CHECK_CUSOLVER(cusolverDnSpotrf(cud_handle_, uplo, n_, d_val_, n_, buffer, bufferSize, info));

  // Eigen::MatrixXf L = Eigen::MatrixXf::Zero(n_, n_);
  // cudaMemcpy(L.data(), d_val_, nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
  // std::cout << Eigen::MatrixXf(L.triangularView<Eigen::Lower>()) << std::endl;

  // typedef Eigen::Matrix<scalar_t, -1, -1> Mat;
  // typedef Eigen::Matrix<scalar_t, -1, 1> Vec;
  
  // Mat A = Mat::Zero(n_, n_);
  // cudaMemcpy(A.data(), d_val_, nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost);  
  // dense_chol(n_, A.data());
  // std::cout << Mat(A.triangularView<Eigen::Upper>()).transpose() << std::endl;

  // Vec b = Vec::Zero(n_);
  // cudaMemcpy(b.data(), d_vecB_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
  // dense_bs(n_, A.data(), b.data());
  // std::cout << b.transpose() << std::endl;
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::compute()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::assemble()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::factorize()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::solve(const scalar_t *h_rhs, scalar_t *h_x)
{
  const scalar_t alpha = 1, beta = 0;

  CHECK_CUDA(cudaMemcpy(d_vecX_, h_rhs, n_*sizeof(scalar_t), cudaMemcpyHostToDevice));
 
  // Note that A_ is of csr format, so no transpose for x = L^T*b
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, A_, x_, &beta, b_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
  // then b = L*x
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                              cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                              &alpha, A_, b_, &beta, x_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));

  CHECK_CUDA(cudaMemcpy(h_x, d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::evalKx(scalar_t *Kx) const
{
  if ( ls_cov_->d_K_val_ ) { // for least square
    typedef typename pde_trait<pde>::scalar_t Scalar;

    // least-square evaluation    
    const Scalar ALPHA = 1, BETA = 0;
    const index_t m = ls_cov_->K_rows_, n = ls_cov_->K_cols_;
    GEMV<Scalar>::run(
        bls_handle_, CUBLAS_OP_N,
        m, n,
        &ALPHA, ls_cov_->d_K_val_, m,
        d_vecX_, 1, &BETA, ls_cov_->d_rhs_bnd_, 1);
    // GEMV<Scalar>::run(
    //     bls_handle_,
    //     cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
    //     m, n,
    //     &ALPHA, ls_cov_->d_K_val_, m,
    //     ls_cov_->d_rhs_bnd_, 1, &BETA, d_vecB_, 1);
    CHECK_CUDA(cudaMemcpy(Kx, ls_cov_->d_rhs_bnd_, m*sizeof(scalar_t), cudaMemcpyDeviceToHost));
  } else {
    const index_t threads_num = 256;
    const index_t blocks_num = (npts_+threads_num-1)/threads_num;
    const index_t sec_blocks_num = (npts_*num_sec_+threads_num-1)/threads_num;

    // eval to tmp    
    CHECK_CUDA(cudaMemset(d_sec_eval_, 0, n_*num_sec_*sizeof(scalar_t)));

    sectioned_eval_Kx<pde, scalar_t, index_t, real_t>
        <<< sec_blocks_num, threads_num >>>
        (npts_, num_sec_, d_xyz_, d_nxyz_, d_ker_aux_, d_vecX_,
         d_sec_eval_, static_cast<index_t>(ker_type_));
    sectioned_eval_reduction(bls_handle_, n_, num_sec_, d_sec_eval_, d_vecB_);
    CHECK_CUDA(cudaMemcpy(Kx, d_vecB_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));    
  }
}

template <enum PDE_TYPE pde>
struct fmm_eval;

template <>
struct fmm_eval<PDE_TYPE::POISSON>
{
  typedef pde_trait<PDE_TYPE::POISSON>::real_t   real_t;
  typedef pde_trait<PDE_TYPE::POISSON>::scalar_t scalar_t;

  // source to source
  static void run(std::vector<std::shared_ptr<H2_3D_Tree>> &tree,
                  std::vector<vector3>  &sources,
                  scalar_t              *weights,
                  scalar_t              *output)
  {
    H2_3D_Compute<H2_3D_Tree> compute
        (*tree[0], sources, sources, weights, 1, output);
  }
  // source to target
  static void run_old(bbfmm3::H2_3D_Tree           *old_tree,
                      std::vector<bbfmm3::vector3> &targets,
                      std::vector<bbfmm3::vector3> &sources,
                      scalar_t                     *weights,
                      int                          ncols,
                      scalar_t                     *output)
  {
    bbfmm3::H2_3D_Compute<bbfmm3::H2_3D_Tree>
        compute(old_tree, &targets[0], &sources[0], sources.size(), targets.size(), weights, ncols, output);
  }
};

template <>
struct fmm_eval<PDE_TYPE::KELVIN>
{
  typedef pde_trait<PDE_TYPE::KELVIN>::real_t   real_t;
  typedef pde_trait<PDE_TYPE::KELVIN>::scalar_t scalar_t;

  // source to source
  static void run(std::vector<std::shared_ptr<H2_3D_Tree>> &tree,
                  std::vector<vector3>  &sources,
                  scalar_t              *weights,
                  scalar_t              *output)
  {
    // weights[ xyz, xyz, xyz, ...] for six PBBFMM which it assumes consecutive storage by channel
    // output [xyz xyz xyz...], stored by point
    const size_t Ns = sources.size(), Nf = Ns;

    // reorder charges in consecutive memory [xxx yyy zzz] and [xxx zzz]
    std::vector<scalar_t> charge_xyz(3*Ns), charge_xz(2*Ns);
    for (size_t i = 0; i < Ns; ++i) {
      charge_xyz[0*Ns+i] = weights[3*i+0];
      charge_xyz[1*Ns+i] = weights[3*i+1];
      charge_xyz[2*Ns+i] = weights[3*i+2];

      charge_xz[0*Ns+i]  = weights[3*i+0];
      charge_xz[1*Ns+i]  = weights[3*i+2];
    }

    std::vector<scalar_t> output0_xx(Nf, 0), output3_yy(Nf, 0), output5_zz(Nf, 0);
    std::vector<scalar_t> output1_xy(2*Nf, 0), output2_xz(2*Nf, 0), output4_yz(2*Nf, 0);

    // xx
    H2_3D_Compute<H2_3D_Tree> compute0
        (*tree[0], sources, sources, &charge_xyz[0*Ns], 1, &output0_xx[0]);
    // xy
    H2_3D_Compute<H2_3D_Tree> compute1
        (*tree[1], sources, sources, &charge_xyz[0*Ns], 2, &output1_xy[0]);
    // xz
    H2_3D_Compute<H2_3D_Tree> compute2
        (*tree[2], sources, sources, &charge_xz[0],     2, &output2_xz[0]);
    // yy
    H2_3D_Compute<H2_3D_Tree> compute3
        (*tree[3], sources, sources, &charge_xyz[1*Ns], 1, &output3_yy[0]);
    // yz
    H2_3D_Compute<H2_3D_Tree> compute4
        (*tree[4], sources, sources, &charge_xyz[1*Ns], 2, &output4_yz[0]);
    // zz
    H2_3D_Compute<H2_3D_Tree> compute5
        (*tree[5], sources, sources, &charge_xyz[2*Ns], 1, &output5_zz[0]);

    // distribute to output
    #pragma omp parallel for
    for (size_t i = 0; i < Nf; ++i) {
      output[3*i+0] = output0_xx[i]+output1_xy[Nf+i]+output2_xz[Nf+i];
      output[3*i+1] = output1_xy[i]+output3_yy[i]+output4_yz[Nf+i];
      output[3*i+2] = output2_xz[i]+output4_yz[i]+output5_zz[i];
    }
  }
  // source to target
  static void run_old(bbfmm3::H2_3D_Tree           *old_tree,
                      std::vector<bbfmm3::vector3> &targets,
                      std::vector<bbfmm3::vector3> &sources,
                      scalar_t                     *weights,
                      int                          ncols,
                      scalar_t                     *output)
  {
    bbfmm3::H2_3D_Compute<bbfmm3::H2_3D_Tree>
        compute(old_tree, &targets[0], &sources[0], sources.size(), targets.size(), weights, ncols, output);
  }
};

template <>
struct fmm_eval<PDE_TYPE::HELMHOLTZ>
{
  typedef pde_trait<PDE_TYPE::HELMHOLTZ>::real_t   real_t;
  typedef pde_trait<PDE_TYPE::HELMHOLTZ>::scalar_t scalar_t;

  // source to source
  static void run(std::vector<std::shared_ptr<H2_3D_Tree>> &tree,
                  std::vector<vector3>  &sources,
                  scalar_t              *weigths,
                  scalar_t              *output)
  {
    // TODO
  }
  // source to target
  static void run_old(bbfmm3::H2_3D_Tree           *old_tree,
                      std::vector<bbfmm3::vector3> &targets,
                      std::vector<bbfmm3::vector3> &sources,
                      scalar_t                     *weights,
                      int                          ncols,
                      scalar_t                     *output)
  {
    // TODO
  }
};

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::evalKx_fmm(std::vector<scalar_t> &Kx,
                                       const FMM_parameters &param)
{
  ASSERT(!ls_cov_->d_K_val_); // not a least-square problem
  
  if ( charges_.size() != n_ ) {
    charges_.resize(n_);
  }
  CHECK_CUDA(cudaMemcpy(&charges_[0], d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));  
    
  std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
  create_fmm_tree(fmm_tree, ker_type_, param);

  // fmm evaluate
  if ( Kx.size() != n_ ) {
    Kx.resize(n_);
  }  
  std::fill(Kx.begin(), Kx.end(), scalar_t(0));    
  fmm_eval<pde>::run(fmm_tree, sources_, &charges_[0], &Kx[0]);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::evalKx_fmm(scalar_t *Kx, const index_t n, const FMM_parameters &param)
{
  ASSERT(!ls_cov_->d_K_val_);  // not a least-square problem
  ASSERT(n_ == n);
  
  if ( charges_.size() != n_ ) {
    charges_.resize(n_);
  }
  CHECK_CUDA(cudaMemcpy(&charges_[0], d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));  
    
  std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
  create_fmm_tree(fmm_tree, ker_type_, param);

  // fmm evaluate
  std::fill(Kx, Kx+n, scalar_t(0));    
  fmm_eval<pde>::run(fmm_tree, sources_, &charges_[0], Kx);
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::pcg_ret_t
gpu_simpl_klchol<pde>::pcg(const scalar_t *h_rhs,
                           scalar_t *h_x,
                           const bool preconditioned,
                           const index_t maxits,
                           const real_t TOL,
                           const FMM_parameters *fmm_param,
                           std::vector<real_t> *residual)
{
  const index_t threads_num = 256;
  const index_t blocks_num = (npts_+threads_num-1)/threads_num;
  const index_t sec_blocks_num = (npts_*num_sec_+threads_num-1)/threads_num;
  const scalar_t ALPHA = 1, BETA = 0;

  if ( residual ) {
    residual->clear();
    residual->reserve(2*maxits);
    residual->emplace_back(1.0);
  }

  // set residual and solution buffer
  CHECK_CUDA(cudaMemcpy(d_resd_, h_rhs, n_*sizeof(scalar_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_*sizeof(scalar_t)));

  scalar_t RHS2 = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      &RHS2,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);
  if ( get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2) < 1e-16 ) {
    std::fill(h_x, h_x+n_, 0);
    return std::make_pair(index_t(0), real_t(0));
  }

  const real_t threshold = TOL*TOL*get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2);

  // solve A*b = resd
  if ( preconditioned ) {
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ALPHA, A_, resd_, &BETA, temp_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                &ALPHA, A_, temp_, &BETA, b_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));    
  } else {
    CHECK_CUDA(cudaMemcpy(d_vecB_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
  }

  scalar_t absNew = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
      &absNew,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);

  index_t i = 0;
  scalar_t RESD2 = 0;

  // cpu buffers for FMM
  if ( potential_.size() != n_ ) {
    potential_.resize(n_);
  }
  if ( charges_.size() != n_ ) {
    charges_.resize(n_);
  }

  while ( i < maxits ) {
    ++i;
    
    // eval to temp, bottelneck of the solver
    if ( ls_cov_->d_K_val_ == NULL ) { // not least-square
      if ( !fmm_param ) {
        // GPU matrix-vector product
        CHECK_CUDA(cudaMemset(d_sec_eval_, 0, n_*num_sec_*sizeof(scalar_t)));
        sectioned_eval_Kx<pde, scalar_t, index_t, real_t>
            <<< sec_blocks_num, threads_num >>>
            (npts_, num_sec_, d_xyz_, d_nxyz_, d_ker_aux_, d_vecB_,
             d_sec_eval_, static_cast<index_t>(ker_type_));
        sectioned_eval_reduction(bls_handle_, n_, num_sec_, d_sec_eval_, d_temp_);
      } else {
        // CPU FMM        
        std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
        create_fmm_tree(fmm_tree, ker_type_, *fmm_param);
        
        // fmm evaluate
        std::fill(potential_.begin(), potential_.end(), scalar_t(0));    
        CHECK_CUDA(cudaMemcpy(&charges_[0], d_vecB_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));    
        fmm_eval<pde>::run(fmm_tree, sources_, &charges_[0], &potential_[0]);
        CHECK_CUDA(cudaMemcpy(d_temp_, &potential_[0], n_*sizeof(scalar_t), cudaMemcpyHostToDevice));
      }
    } else {
      // least-square solve, just two matrix-vector products.. 
      const index_t m = ls_cov_->K_rows_, n = ls_cov_->K_cols_;
      GEMV<scalar_t>::run(
          bls_handle_, CUBLAS_OP_N,
          m, n, 
          &ALPHA, ls_cov_->d_K_val_, m,
          d_vecB_, 1, &BETA, ls_cov_->d_rhs_bnd_, 1);
      GEMV<scalar_t>::run(
          bls_handle_,
          cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
          m, n,
          &ALPHA, ls_cov_->d_K_val_, m,
          ls_cov_->d_rhs_bnd_, 1, &BETA, d_temp_, 1);
    }

    scalar_t b_dot_tmp = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
        d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
        &b_dot_tmp,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    scalar_t alpha = absNew/b_dot_tmp;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecX_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    alpha *= -1;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));

    RESD2 = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        &RESD2,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    const real_t re_resd2 = get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2);
    if ( residual ) {
      residual->emplace_back(re_resd2*TOL*TOL/threshold);
    }
    if ( re_resd2 < threshold ) {
      break;
    }

    // solve A*z = resd
    if ( preconditioned ) {
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &ALPHA, A_, resd_, &BETA, temp_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                  cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                  &ALPHA, A_, temp_, &BETA, z_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    } else {
      CHECK_CUDA(cudaMemcpy(d_z_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
    }

    scalar_t absOld = absNew;
    absNew = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_z_, cuda_data_trait<scalar_t>::dataType, 1,
        &absNew,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);
    
    scalar_t beta = absNew/absOld;
    CHECK_CUBLAS(cublasScalEx(bls_handle_, n_, &beta,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &ALPHA,
                              cuda_data_trait<scalar_t>::dataType,
                              d_z_,    cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
  }
      
  CHECK_CUDA(cudaMemcpy(h_x, d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  return std::make_pair(i, sqrt(
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2)/
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2)));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::thread_safe_pcg_precomp(const FMM_parameters *fmm_param)
{
  if ( fmm_param ) {
    // generate precompute files
    std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
    create_fmm_tree(fmm_tree, ker_type_, *fmm_param);    
  }
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::pcg_ret_t
gpu_simpl_klchol<pde>::thread_safe_pcg(const scalar_t *h_rhs,
                                       scalar_t *h_x,
                                       const bool preconditioned,
                                       const FMM_parameters *fmm_param,
                                       const index_t maxits,
                                       const real_t TOL,
                                       std::vector<real_t> *residual)
{
  // not for least square, for now
  ASSERT(ls_cov_->d_K_val_ == NULL);

  // malloc
  scalar_t *d_resd = NULL, *d_vecX = NULL, *d_vecB = NULL, *d_temp = NULL, *d_vecZ = NULL;
  cusparseDnVecDescr_t resd = NULL, temp = NULL, b = NULL, x = NULL, z = NULL;  
  CHECK_CUDA(cudaMalloc((void **)&d_resd, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_vecX, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_vecB, n_*sizeof(scalar_t)));  
  CHECK_CUDA(cudaMalloc((void **)&d_temp, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_vecZ, n_*sizeof(scalar_t)));   
  CHECK_CUSPARSE(cusparseCreateDnVec(&resd, n_, d_resd, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&temp, n_, d_temp, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&b,    n_, d_vecB, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&x,    n_, d_vecX, cuda_data_trait<scalar_t>::dataType));  
  CHECK_CUSPARSE(cusparseCreateDnVec(&z,    n_, d_vecZ, cuda_data_trait<scalar_t>::dataType));
  
  const scalar_t alpha = 1, beta = 0;
  void *d_work = NULL;
  size_t buff_size = 0;  
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      cus_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_, b, &beta, x,
      cuda_data_trait<scalar_t>::dataType,
      CUSPARSE_SPMV_ALG_DEFAULT,
      &buff_size));    
  CHECK_CUDA(cudaMalloc(&d_work, buff_size));  
  
  const index_t threads_num = 256;
  const index_t blocks_num = (npts_+threads_num-1)/threads_num;
  const index_t sec_blocks_num = (npts_*num_sec_+threads_num-1)/threads_num;
  const scalar_t ALPHA = 1, BETA = 0;

  if ( residual ) {
    residual->clear();
    residual->reserve(2*maxits);
    residual->emplace_back(1.0);
  }

  // set residual and solution buffer
  CHECK_CUDA(cudaMemcpy(d_resd, h_rhs, n_*sizeof(scalar_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_vecX, 0, n_*sizeof(scalar_t)));

  scalar_t RHS2 = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd, cuda_data_trait<scalar_t>::dataType, 1,
      d_resd, cuda_data_trait<scalar_t>::dataType, 1,
      &RHS2,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);
  if ( get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2) < 1e-16 ) {
    std::fill(h_x, h_x+n_, 0);
    return std::make_pair(index_t(0), real_t(0));
  }

  const real_t threshold = TOL*TOL*get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2);

  // solve A*b = resd
  if ( preconditioned ) {
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ALPHA, A_, resd, &BETA, temp,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work));
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                &ALPHA, A_, temp, &BETA, b,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work));    
  } else {
    CHECK_CUDA(cudaMemcpy(d_vecB, d_resd, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
  }

  scalar_t absNew = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd, cuda_data_trait<scalar_t>::dataType, 1,
      d_vecB, cuda_data_trait<scalar_t>::dataType, 1,
      &absNew,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);

  index_t i = 0;
  scalar_t RESD2 = 0;

  // cpu buffers for FMM
  std::vector<scalar_t> potential(n_, 0), charges(n_, 0);

  while ( i < maxits ) {
    ++i;
    
    // CPU FMM        
    std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
    create_fmm_tree(fmm_tree, ker_type_, *fmm_param);
    
    std::fill(potential.begin(), potential.end(), scalar_t(0));
    CHECK_CUDA(cudaMemcpy(&charges[0], d_vecB, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));
    fmm_eval<pde>::run(fmm_tree, sources_, &charges[0], &potential[0]);
    CHECK_CUDA(cudaMemcpy(d_temp, &potential[0], n_*sizeof(scalar_t), cudaMemcpyHostToDevice));

    scalar_t b_dot_tmp = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_vecB, cuda_data_trait<scalar_t>::dataType, 1,
        d_temp, cuda_data_trait<scalar_t>::dataType, 1,
        &b_dot_tmp,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    scalar_t alpha = absNew/b_dot_tmp;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB, cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecX, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    alpha *= -1;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_temp, cuda_data_trait<scalar_t>::dataType, 1,
                              d_resd, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));

    RESD2 = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd, cuda_data_trait<scalar_t>::dataType, 1,
        d_resd, cuda_data_trait<scalar_t>::dataType, 1,
        &RESD2,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    const real_t re_resd2 = get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2);
    if ( residual ) {
      residual->emplace_back(re_resd2*TOL*TOL/threshold);
    }
    if ( re_resd2 < threshold ) {
      break;
    }

    // solve A*z = resd
    if ( preconditioned ) {
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &ALPHA, A_, resd, &BETA, temp,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work));
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                  cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                  &ALPHA, A_, temp, &BETA, z,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work));
    } else {
      CHECK_CUDA(cudaMemcpy(d_vecZ, d_resd, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
    }

    scalar_t absOld = absNew;
    absNew = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd, cuda_data_trait<scalar_t>::dataType, 1,
        d_vecZ, cuda_data_trait<scalar_t>::dataType, 1,
        &absNew,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);
    
    scalar_t beta = absNew/absOld;
    CHECK_CUBLAS(cublasScalEx(bls_handle_, n_, &beta,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &ALPHA,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecZ, cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecB, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
  }
      
  CHECK_CUDA(cudaMemcpy(h_x, d_vecX, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaFree(d_resd));
  CHECK_CUDA(cudaFree(d_vecX));
  CHECK_CUDA(cudaFree(d_vecB));
  CHECK_CUDA(cudaFree(d_vecZ));  
  CHECK_CUDA(cudaFree(d_temp));  
  CHECK_CUSPARSE(cusparseDestroyDnVec(resd));
  CHECK_CUSPARSE(cusparseDestroyDnVec(temp));
  CHECK_CUSPARSE(cusparseDestroyDnVec(b));
  CHECK_CUSPARSE(cusparseDestroyDnVec(x));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
  
  return std::make_pair(i, sqrt(
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2)/
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2)));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::predict(const int impulse_dim,
                                    const scalar_t *src,
                                    const index_t src_num,
                                    scalar_t       *res,
                                    const index_t res_num) const
{
  // storage of impulse, [f_p1, f_p2, ..., f_pm, f_q1, f_q2, ..., f_qm]
  ASSERT(src_num == npts_ && res_num == npts_pred_);

  const int d = pde_trait<pde>::d;
  
  scalar_t *d_src = NULL, *d_res = NULL;
  CHECK_CUDA(cudaMalloc((void **)&d_src, impulse_dim*d*src_num*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_res, impulse_dim*d*res_num*sizeof(scalar_t)));

  // copy input impulses to device
  CHECK_CUDA(cudaMemcpy(d_src, src, impulse_dim*d*src_num*sizeof(scalar_t), cudaMemcpyHostToDevice));

  // extrapolate
  const index_t threads_num = 128; // tunable, not sure if necessary
  const index_t blocks_num = (npts_pred_+threads_num-1)/threads_num;
  index_t shared_mem_size;
  if ( pde != PDE_TYPE::HELMHOLTZ ) {
    shared_mem_size = threads_num*(3+impulse_dim*d)*sizeof(real_t);
  } else {
    shared_mem_size = threads_num*(3+3+2*impulse_dim)*sizeof(real_t);
  }
  tiled_predict<pde>
      <<< blocks_num, threads_num, shared_mem_size >>>
      (npts_pred_, d_pred_xyz_, npts_, d_xyz_, d_nxyz_,
       d_ker_aux_, impulse_dim, d_src, d_res, static_cast<int>(ker_type_));

  // copy back results
  CHECK_CUDA(cudaMemcpy(res, d_res, impulse_dim*d*res_num*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  if ( d_src ) CHECK_CUDA(cudaFree(d_src));
  if ( d_res ) CHECK_CUDA(cudaFree(d_res));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::predict_fmm(const int impulse_dim,
                                        scalar_t *wgt, const index_t wgt_num,
                                        scalar_t *res, const index_t res_num,
                                        const FMM_parameters &param)
{
  // storage of impulse [f_p1, f_q1, ..., f_w1; f_p2, f_q2, ..., f_w2]
  ASSERT(wgt_num == npts_ && res_num == npts_pred_);

  const int d = pde_trait<pde>::d;
  
  // fmm precompute
  std::shared_ptr<bbfmm3::H2_3D_Tree> fmm_tree;
  create_fmm_tree(fmm_tree, ker_type_, param);
  fmm_tree->buildFMMTree();

  // fmm evaluate
  std::fill(res, res+res_num*d*impulse_dim, scalar_t(0));
  fmm_eval<pde>::run_old(fmm_tree.get(), tgts_, srcs_, wgt, impulse_dim, res);
}

/* FOR MERGED MESHES */
template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::train_and_predict(const scalar_t *h_rhs,
                                              const index_t n_train,
                                              scalar_t *h_y,
                                              const index_t threads_num)
{
  ASSERT(n_train <= n_);
  const scalar_t alpha = 1, beta = 0;
  const index_t n_predict = n_-n_train;
  
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_predict*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemcpy(d_vecX_+n_predict, h_rhs, n_train*sizeof(scalar_t), cudaMemcpyHostToDevice));

  // Note that A_ is of csr format, so no transpose for b = L^T*x
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, A_, x_, &beta, b_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
  CHECK_CUDA(cudaMemset(d_vecB_, 0, n_predict*sizeof(scalar_t)));  
  // then x = L*b
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                              cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                              &alpha, A_, b_, &beta, x_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));  

  // // extrapolate
  // const int d = pde_trait<pde>::d;
  // const index_t blocks_num = (n_predict/d+threads_num-1)/threads_num;
  // const index_t sharedMemSize = (3+d)*threads_num*sizeof(real_t);
  // tiled_predict<pde, scalar_t, index_t, real_t>
  //     <<< blocks_num, threads_num, sharedMemSize >>>
  //     (n_predict/d, n_train/d, d_xyz_, d_ker_aux_, d_vecX_+n_predict,
  //      d_temp_, static_cast<index_t>(ker_type_));

  // CHECK_CUDA(cudaMemcpy(h_y, d_temp_, n_predict*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  // cudaDeviceSynchronize(); // synchronize for counting wall-clock time  
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::chol_predict(const scalar_t *h_train,
                                         const index_t n_train,
                                         scalar_t *h_pred)
{
  bool allow_chol_predict = false;
  ASSERT(allow_chol_predict);

#if 0
  ASSERT(n_train <= n_);
  
  scalar_t alpha = 1, beta = 0;
  const index_t n_predict = n_-n_train;
  
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_predict*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemcpy(d_vecX_+n_predict, h_train, n_train*sizeof(scalar_t), cudaMemcpyHostToDevice));

  // Note that A_ is of csr format, so no transpose for b = L^T*x
  CHECK_CUSPARSE(cusparseSpMV(
      cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_, x_, &beta, b_,
      cuda_data_trait<scalar_t>::dataType,
      CUSPARSE_SPMV_ALG_DEFAULT,
      d_work_));
  CHECK_CUDA(cudaMemset(d_vecB_+n_predict, 0, n_train*sizeof(scalar_t)));

  // upper triangular solve to x
  alpha = -1;
  CHECK_CUSPARSE(cusparseSpSV_analysis(
      cus_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_, b_, x_,
      cuda_data_trait<scalar_t>::dataType,
      CUSPARSE_SPSV_ALG_DEFAULT,
      spsv_desc_,
      d_sv_work_));  
  CHECK_CUSPARSE(cusparseSpSV_solve(
      cus_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_, b_, x_,
      cuda_data_trait<scalar_t>::dataType,
      CUSPARSE_SPSV_ALG_DEFAULT,
      spsv_desc_));

  CHECK_CUDA(cudaMemcpy(h_pred, d_vecX_, n_predict*sizeof(scalar_t), cudaMemcpyDeviceToHost));
#endif
}
//==============================================================
template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::gpu_super_klchol(const index_t npts, const index_t ker_dim, const index_t sec_num)
    : gpu_simpl_klchol<pde>(npts, ker_dim, sec_num)
{
}

template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::~gpu_super_klchol()
{
  if ( d_super_ptr_    ) { CHECK_CUDA(cudaFree(d_super_ptr_));    }
  if ( d_super_ind_    ) { CHECK_CUDA(cudaFree(d_super_ind_));    }
  if ( d_super_parent_ ) { CHECK_CUDA(cudaFree(d_super_parent_)); }

  if ( h_super_ptr_ ) { delete[] h_super_ptr_; }
  if ( h_super_ind_ ) { delete[] h_super_ind_; }
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::set_supernodes(const index_t n_super,
                                      const index_t n_pts,
                                      const index_t *h_super_ptr,
                                      const index_t *h_super_ind,
                                      const index_t *h_super_parent)
{
  ASSERT(this->npts_ == n_pts);
  ASSERT(*std::max_element(h_super_parent, h_super_parent+n_pts)+1 == n_super);

  n_super_ = n_super;
  spdlog::info("n_super_={}, npts_={}", n_super_, this->npts_);

  // store a copy on cpu
  h_super_ptr_ = new index_t[n_super_+1];
  h_super_ind_ = new index_t[this->npts_];
  std::copy(h_super_ptr, h_super_ptr+n_super_+1, h_super_ptr_);
  std::copy(h_super_ind, h_super_ind+this->npts_, h_super_ind_);

  // gpu memory allocation
  CHECK_CUDA(cudaMalloc((void **)&d_super_ptr_,    (n_super_+1)*sizeof(index_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_super_ind_,    this->npts_*sizeof(index_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_super_parent_, this->npts_*sizeof(index_t)));

  CHECK_CUDA(cudaMemcpy(d_super_ptr_, h_super_ptr,    (n_super_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_super_ind_, h_super_ind,    this->npts_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_super_parent_, h_super_parent, this->npts_*sizeof(index_t), cudaMemcpyHostToDevice));
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::set_sppatt(const index_t n,
                                  const index_t nnz,
                                  const index_t *h_ptr,
                                  const index_t *h_ind)
{
  ASSERT(n == this->n_ && this->npts_*this->ker_dim_ == n);
  ASSERT(d_super_ptr_ != NULL && d_super_ind_ != NULL);

  const auto &malloc_factor =
      [&]() {
        CHECK_CUDA(cudaMalloc((void **)&this->d_ptr_, (this->n_+1)*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_ind_, this->nnz_*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_val_, this->nnz_*sizeof(scalar_t)));
        CHECK_CUSPARSE(cusparseCreateCsr(&this->A_, this->n_, this->n_, this->nnz_, this->d_ptr_, this->d_ind_, this->d_val_,
                                         cuda_index_trait<index_t>::indexType,
                                         cuda_index_trait<index_t>::indexType,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_data_trait<scalar_t>::dataType));

        // cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
        // CHECK_CUSPARSE(cusparseSpMatSetAttribute(this->A_, CUSPARSE_SPMAT_FILL_MODE,
        //                                          &fillmode, sizeof(fillmode)));
        // cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
        // CHECK_CUSPARSE(cusparseSpMatSetAttribute(this->A_, CUSPARSE_SPMAT_DIAG_TYPE,
        //                                          &diagtype, sizeof(diagtype)));
      };

  const auto &malloc_theta = 
      [&]() {
        std::vector<index_t> ptr(n_super_+1, 0);
        for (index_t j = 0; j < n_super_; ++j) {
          // for each super node
          const index_t super_iter = h_super_ptr_[j];
          const index_t first_dof = this->ker_dim_*h_super_ind_[super_iter];
          const index_t nnz_super_j = h_ptr[first_dof+1]-h_ptr[first_dof];
          ptr[j+1] = ptr[j]+nnz_super_j*nnz_super_j;
        }
        this->TH_nnz_ = ptr.back();
        const real_t Gb = this->TH_nnz_*sizeof(scalar_t)/(1024*1024*1024);
        spdlog::info("total size for THETA={0:.1f}, {1:.2f} GB", this->TH_nnz_, Gb);
        ASSERT(this->TH_nnz_ < INT_MAX);        

        CHECK_CUDA(cudaMalloc((void **)&this->d_TH_ptr_, ptr.size()*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_TH_val_, ptr.back()*sizeof(scalar_t)));
        CHECK_CUDA(cudaMemcpy(this->d_TH_ptr_, &ptr[0], ptr.size()*sizeof(index_t), cudaMemcpyHostToDevice));
      };

  const auto &malloc_mv_buffer =
      [&]() {
        const scalar_t alpha = 1, beta = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            this->cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, this->A_, this->b_, &beta, this->x_,
            cuda_data_trait<scalar_t>::dataType,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &this->buff_sz_));    
        CHECK_CUDA(cudaMalloc(&this->d_work_, this->buff_sz_));
        spdlog::info("SpMV buffer size={}", this->buff_sz_);
      };

  const auto &malloc_sv_buffer =
      [&]() {
        // const scalar_t alpha = 1;
        // CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        //     this->cus_handle_,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     this->A_, this->b_, this->x_,
        //     cuda_data_trait<scalar_t>::dataType,
        //     CUSPARSE_SPSV_ALG_DEFAULT,
        //     this->spsv_desc_,
        //     &this->sv_buff_sz_));
        // CHECK_CUDA(cudaMalloc(&this->d_sv_work_, this->sv_buff_sz_));
        // spdlog::info("SpSV buffer size={}", this->sv_buff_sz_);
      };

  if ( this->d_ptr_ == NULL ) { // allocate for the first time
    this->nnz_ = nnz;

    malloc_factor();
    malloc_theta();
    malloc_mv_buffer();
    malloc_sv_buffer();
  }

  if ( this->d_ptr_ && this->nnz_ != nnz ) { // reallocate if nnz is changed
    this->nnz_ = nnz;

    CHECK_CUDA(cudaFree(this->d_ptr_));
    CHECK_CUDA(cudaFree(this->d_ind_));
    CHECK_CUDA(cudaFree(this->d_val_));
    CHECK_CUSPARSE(cusparseDestroySpMat(this->A_));
    malloc_factor();

    CHECK_CUDA(cudaFree(this->d_TH_ptr_));
    CHECK_CUDA(cudaFree(this->d_TH_val_));
    malloc_theta();

    CHECK_CUDA(cudaFree(this->d_work_));
    malloc_mv_buffer();

    // CHECK_CUDA(cudaFree(this->d_sv_work_));
    malloc_sv_buffer();
  }

  // copy factor sparsity
  CHECK_CUDA(cudaMemcpy(this->d_ptr_, h_ptr, (this->n_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(this->d_ind_, h_ind, this->nnz_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));
}

template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::real_t gpu_super_klchol<pde>::memory() const {
  const real_t total_bytes = 
      (this->n_+1)*sizeof(index_t)+
      this->nnz_*sizeof(index_t)+
      this->nnz_*sizeof(scalar_t)+
      (n_super_+1)*sizeof(index_t)+
      this->TH_nnz_*sizeof(scalar_t);  
  return total_bytes/1024/1024/1024;
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::compute()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));

  // assemble
  if ( this->ker_dim_ == 1 ) {
    klchol_super_fac_asm_scalar<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  } else if ( this->ker_dim_ == 3 ) {
    klchol_super_fac_asm_vector<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  }

#if 0
  {
  cudaDeviceSynchronize();  
  Eigen::VectorXi th_ptr(n_super_+1);
  cudaMemcpy(th_ptr.data(), this->d_TH_ptr_, (n_super_+1)*sizeof(index_t), cudaMemcpyDeviceToHost);
  Eigen::VectorXcd th_val(static_cast<index_t>(this->TH_nnz_));
  cudaMemcpy(th_val.data(), this->d_TH_val_, this->TH_nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
  #pragma omp parallel for
  for (size_t j = 0; j < th_ptr.size()-1; ++j) {
    const size_t local_n = floor(sqrt(1.0*(th_ptr[j+1]-th_ptr[j]))+0.5);
    Eigen::MatrixXcd tmp = th_val.segment(th_ptr[j], local_n*local_n).reshaped(local_n, local_n);
    Eigen::MatrixXcd local_TH = Eigen::SelfAdjointView<Eigen::MatrixXcd, Eigen::Upper>(tmp);
    // if ( j < 3 ) {
    //   std::cout << local_TH << std::endl << std::endl;
    // }
    Eigen::LLT<Eigen::MatrixXcd> llt;
    llt.compute(local_TH);
    if ( llt.info() != Eigen::Success && j < 100 ) {
      std::cout << j << std::endl;
      std::cout << local_n << std::endl;
      std::cout << local_TH.topLeftCorner(5, 5) << std::endl << std::endl;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig;
      eig.compute(local_TH);
      std::cout << eig.eigenvalues().transpose() << std::endl;
    }
  }
  }
#endif

  klchol_super_fac_chol<<< blocks_super, threads_num >>>
      (n_super_, this->d_TH_ptr_, this->d_TH_val_);
#if 0
  {
  cudaDeviceSynchronize();  
  Eigen::VectorXi th_ptr(n_super_+1);
  cudaMemcpy(th_ptr.data(), this->d_TH_ptr_, (n_super_+1)*sizeof(index_t), cudaMemcpyDeviceToHost);
  Eigen::VectorXcd th_val(static_cast<index_t>(this->TH_nnz_));
  cudaMemcpy(th_val.data(), this->d_TH_val_, this->TH_nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
  #pragma omp parallel for
  for (size_t j = 0; j < th_ptr.size()-1; ++j) {
    const size_t local_n = floor(sqrt(1.0*(th_ptr[j+1]-th_ptr[j]))+0.5);
    Eigen::MatrixXcd local_TH = th_val.segment(th_ptr[j], local_n*local_n).reshaped(local_n, local_n);
    const Eigen::MatrixXcd U = local_TH.triangularView<Eigen::Upper>();
    // if ( j < 3 ) {
    //   std::cout << U.adjoint()*U << std::endl << std::endl;
    // }

    Eigen::LLT<Eigen::MatrixXcd> llt;
    llt.compute(local_TH);
    if ( llt.info() != Eigen::Success ) {
      std::cout << j << std::endl;
      std::cout << local_n << std::endl;
      std::cout << local_TH.topLeftCorner(5, 5) << std::endl << std::endl;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig;
      eig.compute(local_TH);
      std::cout << eig.eigenvalues().head(10).transpose() << std::endl;
    }
  }
  }
#endif
  
  klchol_super_fac_bs<<< blocks_dof, threads_num >>>
      (this->n_, this->ker_dim_, this->d_ptr_, this->d_ind_, this->d_val_,
       d_super_parent_, this->d_TH_ptr_, this->d_TH_val_);
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::assemble()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));

  if ( this->ker_dim_ == 1 ) {
    klchol_super_fac_asm_scalar<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  } else if ( this->ker_dim_ == 3 ) {
    klchol_super_fac_asm_vector<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  }
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::factorize()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  klchol_super_fac_chol<<< blocks_super, threads_num >>>
      (n_super_, this->d_TH_ptr_, this->d_TH_val_);
  
  klchol_super_fac_bs<<< blocks_dof, threads_num >>>
      (this->n_, this->ker_dim_, this->d_ptr_, this->d_ind_, this->d_val_,
       d_super_parent_, this->d_TH_ptr_, this->d_TH_val_);
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::debug(const scalar_t *aux, const int n)
{  
  // using namespace Eigen;
  // using namespace std;
  // std::cout << "debug" << std::endl;
  
  // Eigen::VectorXi ptr(this->n_+1), ind(this->nnz_);
  // Eigen::Matrix<scalar_t, -1, 1> val(this->nnz_);
  // CHECK_CUDA(cudaMemcpy(ptr.data(), this->d_ptr_, (this->n_+1)*sizeof(int), cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(ind.data(), this->d_ind_, (this->nnz_)*sizeof(int), cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(val.data(), this->d_val_, (this->nnz_)*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  // Eigen::Matrix<scalar_t, -1, -1> L(this->n_, this->n_);
  // L.setZero();
  // for (int j = 0; j < this->n_; ++j) {
  //   for (int iter = ptr[j]; iter < ptr[j+1]; ++iter) {
  //     L(ind[iter], j) = val[iter];
  //   }
  // }

  // Eigen::Matrix<double, -1, -1, Eigen::RowMajor> pts(this->npts_, 3);
  // cudaMemcpy(pts.data(), this->d_xyz_, 3*this->npts_*sizeof(double), cudaMemcpyDeviceToHost);

  // Eigen::VectorXd param(8);
  // cudaMemcpy(param.data(), this->d_ker_aux_, 8*sizeof(double), cudaMemcpyDeviceToHost);
  
  // Eigen::Matrix<scalar_t, -1, -1> TH(this->n_, this->n_);
  // for (size_t i = 0; i < this->npts_; ++i) {
  //   for (size_t j = 0; j < this->npts_; ++j) {
  //     gf_summary<pde, scalar_t, real_t>::run(static_cast<int>(this->ker_type_), &pts(i, 0), &pts(j, 0), param.data(), &TH(i, j));
  //   }
  // }

  // // std::cout << TH << std::endl;
  // std::cout << (TH.inverse()-L*L.transpose()).norm()/TH.inverse().norm() << std::endl;
  // std::cout << (TH-L.transpose().inverse()*L.inverse()).norm()/TH.norm() << std::endl;

  // int np = this->n_-n;
  // VectorXd rhs(n);
  // std::copy(aux, aux+n, rhs.data());
  // MatrixXd L_b = L.bottomRightCorner(n, n);
  // VectorXd sol = L_b*L_b.transpose()*rhs;
  // spdlog::info("solnorm={0:.8f}", sol.norm());
  // VectorXd qsol = TH.block(0, np, np, n)*sol;
  // VectorXd psol = -L.topLeftCorner(np, np).transpose().inverse()*
  //     L.block(np, 0, n, np).transpose()*rhs;
  // cout << (qsol-psol).norm()/qsol.norm() << endl;
  // cout << "first pass norm=" << (-L.topLeftCorner(np, np).transpose().inverse()*L.block(np, 0, n, np).transpose()*rhs).transpose() << endl;
  // getchar();
  
  // srand(time(NULL));
  // Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
  // A = (A.transpose()*A).eval();
  // dense_chol(A.rows(), A.data());
  // getchar();
  
  // int rv_size[3] = {8, 9, 10};
  // for (int i = 0; i < 3; ++i) {
  //   Eigen::VectorXd rv = Eigen::VectorXd::Random(rv_size[i]);
  //   std::cout << rv.transpose() << std::endl;
  //   reverse_inplace(rv.size(), rv.data());
  //   std::cout << rv.transpose() << std::endl << std::endl;
  // }

  // srand(time(NULL));
  // const size_t N = 10, subN = 5;
  // const Eigen::MatrixXd L = Eigen::MatrixXd::Random(N, N);
  // Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N);
  // rhs[subN-1] = 1;
  // Eigen::VectorXd res = L.triangularView<Eigen::Upper>().solve(rhs);
  // std::cout << "res full=" << res.transpose() << std::endl;

  // {
  //   Eigen::VectorXd res_2 = rhs.head(subN);
  //   upper_tri_solve(L.rows(), L.data(), res_2.size(), res_2.data());
  //   std::cout << "res part=" << res_2.transpose() << std::endl;
  // }
}

// explicit instanialization
template class gpu_simpl_klchol<PDE_TYPE::POISSON>;
template class gpu_super_klchol<PDE_TYPE::POISSON>;

template class gpu_simpl_klchol<PDE_TYPE::KELVIN>;
template class gpu_super_klchol<PDE_TYPE::KELVIN>;

template class gpu_simpl_klchol<PDE_TYPE::HELMHOLTZ>;
template class gpu_super_klchol<PDE_TYPE::HELMHOLTZ>;

template struct cov_assembler<PDE_TYPE::POISSON>;
template struct cov_assembler<PDE_TYPE::KELVIN>;
template struct cov_assembler<PDE_TYPE::HELMHOLTZ>;

}
