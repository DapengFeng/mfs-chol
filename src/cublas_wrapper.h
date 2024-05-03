#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <cublas_v2.h>

namespace klchol {

/// ---------------------- dot product ---------------------------
template <typename scalar_t>
struct DOTP;

template <>
struct DOTP<double>
{
  __forceinline__
  static void run(cublasHandle_t handle,
                  int n,
                  const void *x,
                  cudaDataType xType,
                  int incx,
                  const void *y,
                  cudaDataType yType,
                  int incy,
                  void *result,
                  cudaDataType resultType,
                  cudaDataType executionType)
  {
    CHECK_CUBLAS(cublasDotEx(
        handle,
        n,
        x,
        xType,
        incx,
        y,
        yType,
        incy,
        result,
        resultType,
        executionType));
  }  
};

template <>
struct DOTP<thrust::complex<double>>
{
  __forceinline__
  static void run(cublasHandle_t handle,
                  int n,
                  const void *x,
                  cudaDataType xType,
                  int incx,
                  const void *y,
                  cudaDataType yType,
                  int incy,
                  void *result,
                  cudaDataType resultType,
                  cudaDataType executionType)
  {
    CHECK_CUBLAS(cublasDotcEx(
        handle,
        n,
        x,
        xType,
        incx,
        y,
        yType,
        incy,
        result,
        resultType,
        executionType));
  }
};

/// ------------------- matrix-vector product --------------------
template <typename scalar_t>
struct GEMV;

template <>
struct GEMV<double>
{
  __forceinline__
  static void run(cublasHandle_t handle, cublasOperation_t trans,
                  int m, int n,
                  const double          *alpha,
                  const double          *A, int lda,
                  const double          *x, int incx,
                  const double          *beta,
                  double          *y, int incy)
  {
    CHECK_CUBLAS(cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
  }
};

template <>
struct GEMV<thrust::complex<double>>
{
  __forceinline__
  static void run(cublasHandle_t handle, cublasOperation_t trans,
                  int m, int n,
                  const thrust::complex<double> *alpha,
                  const thrust::complex<double> *A, int lda,
                  const thrust::complex<double> *x, int incx,
                  const thrust::complex<double> *beta,
                  thrust::complex<double> *y, int incy)
  {
    CHECK_CUBLAS(cublasZgemv(handle, trans, m, n,
                             reinterpret_cast<const cuDoubleComplex*>(alpha),
                             reinterpret_cast<const cuDoubleComplex*>(A), lda,
                             reinterpret_cast<const cuDoubleComplex*>(x), incx,
                             reinterpret_cast<const cuDoubleComplex*>(beta),
                             reinterpret_cast<cuDoubleComplex*>(y), incy));
  }
};

}
#endif
