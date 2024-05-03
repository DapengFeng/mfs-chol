#ifndef KL_KERNELS_H
#define KL_KERNELS_H

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <bbfmm3d.hpp>
#include <old_bbfmm3d.hpp>
#include <memory>

#include "traits.h"

namespace klchol {

#define CUDA_PI 3.1415926535897932
#define SQRT_3  1.7320508075688772
#define R_4PI   0.07957747154594767

// -------------------- kernel functions ----------------------------
enum KERNEL_TYPE
{
  MATERN = 0,
  LAPLACE_2D,  
  LAPLACE_3D,
  SCR_POISSON_2D,  
  SCR_POISSON_3D,
  KELVIN_3D,
  HELMHOLTZ_2D,
  HELMHOLTZ_3D,
  BILAP_REG_3D,
  LAPLACE_NM_2D,
  HELMHOLTZ_NM_2D
};

const char *KERNEL_LIST = "matern\0lap_2d\0lap_3d\0scr_poi_2d\0scr_poi_3d\0kelvin_3d\0helm_2d\0helm_3d\0bilap_reg_3d\0lap_nm_2d\0helm_nm_2d\0\0";

template <typename scalar_t>
__device__ __host__
static scalar_t bessi0(scalar_t x)
{
   scalar_t ax,ans;
   scalar_t y;

   if ((ax=fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
         +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
   } else {
      y=3.75/ax;
      ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
         +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
         +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
         +y*0.392377e-2))))))));
   }
   return ans;
}

template <typename scalar_t>
__device__ __host__
static scalar_t bessk0(scalar_t x)
{
   scalar_t y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;  
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void matern_cov(const real_t *x,
                const real_t *y,
                const real_t *n_x,
                const real_t *n_y,
                const real_t *p,
                scalar_t     *G)
{
  // nu = 3/2
  const scalar_t l = p[0];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r_over_l = sqrt(3*(dx*dx+dy*dy+dz*dz))/l;
  *G = (1+r_over_l)*exp(-r_over_l);
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void laplace_3d(const real_t *x,
                const real_t *y,
                const real_t *n_x,
                const real_t *n_y,                
                const real_t *p,
                scalar_t     *G)
{
  const scalar_t eps2 = p[0]*p[0];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r2 = dx*dx+dy*dy+dz*dz;
  const scalar_t reg_r = sqrt(r2+eps2);
  *G = R_4PI/reg_r;
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void laplace_2d(const real_t *x,
                const real_t *y,
                const real_t *n_x,
                const real_t *n_y,
                const real_t *p,
                scalar_t     *G)
{
  const scalar_t eps2 = p[0]*p[0];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r2 = dx*dx+dy*dy+dz*dz+eps2;
  *G = -R_4PI*log(r2);
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void laplace_nm_2d(const real_t *x,
                   const real_t *y,
                   const real_t *n_x,
                   const real_t *n_y,
                   const real_t *p,
                   scalar_t     *G)
{
  const real_t eps2 = p[0]*p[0];
  const real_t r[3] = {*x-*y, *(x+1)-*(y+1), *(x+2)-*(y+2)};

  const real_t reg_r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + eps2,
      reg_r1 = sqrt(reg_r2), reg_r3 = reg_r1*reg_r2;
  real_t rnx = r[0]*n_x[0] + r[1]*n_x[1] + r[2]*n_x[2];
  real_t rny = r[0]*n_y[0] + r[1]*n_y[1] + r[2]*n_y[2];
  real_t nxny = n_x[0]*n_y[0] + n_x[1]*n_y[1] + n_x[2]*n_y[2];

  scalar_t gra = -2*R_4PI/reg_r1, hes = 2*R_4PI/reg_r2;
  *G = (gra/reg_r3-hes/reg_r2)*rnx*rny - gra/reg_r1*nxny;
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void kelvin_3d(const real_t *x,
               const real_t *y,
               const real_t *n_x,
               const real_t *n_y,               
               const real_t *p,
               scalar_t     *K)
{
  typedef Eigen::Matrix<scalar_t, 3, 3> Mat3f;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vec3f;  
  
  const scalar_t eps2 = p[0]*p[0], a = p[1], b = p[2];
  Vec3f r;
  r[0] = *(x)-*(y);
  r[1] = *(x+1)-*(y+1);
  r[2] = *(x+2)-*(y+2);
  const scalar_t rnorm = sqrt(r.squaredNorm()+eps2), rnorm3 = rnorm*rnorm*rnorm;
  // Eigen::Map<Mat3f>(K, 3, 3) =
  //     ((a-b)/rnorm+0.5*a*eps2/rnorm3)*Mat3f::Identity()+
  //     b/rnorm3*r*r.transpose();
  Eigen::Map<Mat3f>(K, 3, 3) =
      (a-b)/rnorm*Mat3f::Identity() + b/rnorm3*r*r.transpose();  
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void helmholtz_2d(const real_t *x,
                  const real_t *y,
                  const real_t *n_x,
                  const real_t *n_y,                  
                  const real_t *p,
                  scalar_t     *H)
{
  const real_t eps2 = p[0]*p[0], K = p[1];
  const real_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const real_t r2 = dx*dx+dy*dy+dz*dz+eps2;
  const real_t kr = K*sqrt(r2);
  const scalar_t I(0.0, 1.0);
  *H = 0.25*I*(j0(kr)+I*y0(kr));
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void helmholtz_nm_2d(const real_t *x,
                     const real_t *y,
                     const real_t *n_x,
                     const real_t *n_y,                  
                     const real_t *p,
                     scalar_t     *H)
{
  const real_t eps2 = p[0]*p[0], K = p[1], alpha = p[6], beta = p[7];
  const real_t r[3] = {*x-*y, *(x+1)-*(y+1), *(x+2)-*(y+2)};

  const real_t r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + eps2,
      r1 = sqrt(r2), r3 = r1*r2;
  real_t rnx = r[0]*n_x[0] + r[1]*n_x[1] + r[2]*n_x[2];
  real_t rny = r[0]*n_y[0] + r[1]*n_y[1] + r[2]*n_y[2];
  real_t nxny = n_x[0]*n_y[0] + n_x[1]*n_y[1] + n_x[2]*n_y[2];
  real_t kr = K*r1;

  const scalar_t I(0.0, 1.0);
  scalar_t gra = -0.25*I*K*(j1(kr)+I*y1(kr)),
      hes = -0.25*I*K*K*( (j0(kr)-j1(kr)/kr) + I*(y0(kr)-y1(kr)/kr) );
  *H = alpha*(gra/r3-hes/r2)*rnx*rny - beta*gra/r1*nxny;
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void helmholtz_3d(const real_t *x,
                  const real_t *y,
                  const real_t *n_x,
                  const real_t *n_y,
                  const real_t *p,
                  scalar_t     *H)
{
  const real_t eps2 = p[0]*p[0], K = p[1];
  const real_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const real_t r2 = dx*dx+dy*dy+dz*dz + eps2;
  const real_t reg_r = sqrt(r2);
  *H = R_4PI*exp(scalar_t(0.0, 1.0)*K*reg_r)/reg_r;
}

template <typename scalar_t, typename real_t>
__device__ __host__
void screen_poisson_2d(const real_t *x,
                       const real_t *y,
                       const real_t *n_x,
                       const real_t *n_y,
                       const real_t *p,
                       scalar_t     *H)
{
  const scalar_t eps2 = p[0]*p[0], K = p[1];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r2 = dx*dx+dy*dy+dz*dz;
  const scalar_t reg_r = sqrt(r2+eps2);
  *H = 2*R_4PI*bessk0(K*reg_r);
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void screen_poisson_3d(const real_t *x,
                       const real_t *y,
                       const real_t *n_x,
                       const real_t *n_y,
                       const real_t *p,
                       scalar_t     *H)
{
  const scalar_t eps2 = p[0]*p[0], K = p[1];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r2 = dx*dx+dy*dy+dz*dz;
  const scalar_t reg_r = sqrt(r2+eps2);
  *H = R_4PI*exp(-K*sqrt(r2))/reg_r;
}

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void bilap_reg_3d(const real_t *x,
                  const real_t *y,
                  const real_t *n_x,
                  const real_t *n_y,                  
                  const real_t *p,
                  scalar_t     *H)
{
  const scalar_t eps = p[0];
  const scalar_t dx = *x-*y, dy = *(x+1)-*(y+1), dz = *(x+2)-*(y+2);
  const scalar_t r = sqrt(dx*dx+dy*dy+dz*dz);
  *H = r < 1e-10 ? R_4PI/eps : R_4PI*(1-exp(-r/eps))/r;
}

// ------------------- BBFMM kernels --------------------------------------
struct FMM_parameters
{
  double L                = 1.0;
  int tree_level          = 4;
  int interpolation_order = 3;
  double eps              = 1e-4;
  int use_chebyshev       = 1;
  double reg_eps          = 0;

  // for screened Poisson
  double K                = 0;

  // for kelvinlet
  double a                = 0;
  double b                = 0;
};

// ================== PBBFMM3D ====================
class PBBFMM_laplace_2d : public H2_3D_Tree
{
 public:
  const double reg_eps_;

  PBBFMM_laplace_2d(double L, int tree_level,
                    int interpolation_order, double eps,
                    int use_chebyshev, const double reg_eps)
      : reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
    std::cout << "[PBBFMM_laplace2d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_laplace_2d";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r2 = dx*dx+dy*dy+dz*dz+reg_eps_*reg_eps_;
    return -R_4PI*log(r2);
  }
};

class PBBFMM_laplace_3d : public H2_3D_Tree
{
 public:
  const double reg_eps_;

  PBBFMM_laplace_3d(double L, int tree_level,
                    int interpolation_order, double eps,
                    int use_chebyshev, const double reg_eps)
      : reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
    std::cout << "[PBBFMM_laplace3d] reg_eps_=" << reg_eps_ << std::endl;    
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_laplace_3d";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r2 = dx*dx+dy*dy+dz*dz+reg_eps_*reg_eps_;
    return R_4PI/sqrt(r2);
  }
};

class PBBFMM_scr_poisson_2d : public H2_3D_Tree
{
 public:
  const double reg_eps_, K_;

  PBBFMM_scr_poisson_2d(double L, int tree_level,
                        int interpolation_order, double eps,
                        int use_chebyshev, const double reg_eps,
                        const double K)
      : reg_eps_(reg_eps), K_(K), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
    std::cout << "[PBBFMM_scr_poisson2d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_scr_poisson_2d";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double reg_r = sqrt(dx*dx+dy*dy+dz*dz+reg_eps_*reg_eps_);
    return 2*R_4PI*bessk0(K_*reg_r);
  }
};

class PBBFMM_bilaplap_3d : public H2_3D_Tree
{
 public:
  const double reg_eps_;

  PBBFMM_bilaplap_3d(double L, int tree_level,
                     int interpolation_order, double eps,
                     int use_chebyshev, const double reg_eps)
      : reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
    std::cout << "[PBBFMM_bilaplap3d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_bilaplap_3d";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r  = sqrt(dx*dx+dy*dy+dz*dz);
    return r < 1e-10 ? R_4PI/reg_eps_ : R_4PI*(1-exp(-r/reg_eps_))/r;
  }
};

class PBBFMM_kelvin_3d_xx : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_xx(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_xx";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return (a_-b_)/reg_r1 + b_/(reg_r1*reg_r2)*r[0]*r[0];
  }  
};

class PBBFMM_kelvin_3d_yy : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_yy(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_yy";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return (a_-b_)/reg_r1 + b_/(reg_r1*reg_r2)*r[1]*r[1];
  }  
};

class PBBFMM_kelvin_3d_zz : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_zz(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_zz";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return (a_-b_)/reg_r1 + b_/(reg_r1*reg_r2)*r[2]*r[2];
  }  
};

class PBBFMM_kelvin_3d_xy : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_xy(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_xy";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return b_/(reg_r1*reg_r2)*r[0]*r[1];
  }  
};

class PBBFMM_kelvin_3d_xz : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_xz(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_xz";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return b_/(reg_r1*reg_r2)*r[0]*r[2];
  }  
};

class PBBFMM_kelvin_3d_yz : public H2_3D_Tree
{
public:
  const double reg_eps_, a_, b_;

  PBBFMM_kelvin_3d_yz(double L,
                      int tree_level,
                      int interpolation_order,
                      double eps,
                      int use_chebyshev,
                      const double a,
                      const double b,
                      const double reg_eps)
      : a_(a), b_(b), reg_eps_(reg_eps), H2_3D_Tree(L, tree_level, interpolation_order, eps, use_chebyshev)
  {
  }
  virtual void SetKernelProperty()
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "PBBFMM_kelvin_3d_yz";
  }
  virtual double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
  {
    const double r[3] = {
      sourcepos.x - targetpos.x,
      sourcepos.y - targetpos.y,
      sourcepos.z - targetpos.z
    };
    const double reg_r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+reg_eps_*reg_eps_;
    const double reg_r1 = sqrt(reg_r2);
    return b_/(reg_r1*reg_r2)*r[1]*r[2];
  }  
};

void create_fmm_tree(std::vector<std::shared_ptr<H2_3D_Tree>> &trees,
                     const KERNEL_TYPE kernel,
                     const FMM_parameters &param)
{
  if ( kernel == LAPLACE_2D ) {
    trees.emplace_back(
        std::make_shared<PBBFMM_laplace_2d>
        (param.L, param.tree_level, param.interpolation_order,
         param.eps, param.use_chebyshev, param.reg_eps));
    trees.back()->buildFMMTree();
  } else if ( kernel == LAPLACE_3D ) {
    trees.emplace_back(
        std::make_shared<PBBFMM_laplace_3d>
        (param.L, param.tree_level, param.interpolation_order,
         param.eps, param.use_chebyshev, param.reg_eps));
    trees.back()->buildFMMTree();
  } else if ( kernel == BILAP_REG_3D ) {
    trees.emplace_back(
        std::make_shared<PBBFMM_bilaplap_3d>
        (param.L, param.tree_level, param.interpolation_order,
         param.eps, param.use_chebyshev, param.reg_eps));
    trees.back()->buildFMMTree();
  } else if ( kernel == SCR_POISSON_2D ) {
    trees.emplace_back(
        std::make_shared<PBBFMM_scr_poisson_2d>
        (param.L, param.tree_level, param.interpolation_order,
         param.eps, param.use_chebyshev, param.reg_eps, param.K));
    trees.back()->buildFMMTree();
  } else if ( kernel == KELVIN_3D ) {
    trees.resize(6);
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        trees[0] = std::make_shared<PBBFMM_kelvin_3d_xx>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[0]->buildFMMTree();
      }
      #pragma omp section
      {
        trees[1] = std::make_shared<PBBFMM_kelvin_3d_xy>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[1]->buildFMMTree();
      }
      #pragma omp section
      {
        trees[2] = std::make_shared<PBBFMM_kelvin_3d_xz>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[2]->buildFMMTree();            
      }
      #pragma omp section
      {
        trees[3] = std::make_shared<PBBFMM_kelvin_3d_yy>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[3]->buildFMMTree();            
      }
      #pragma omp section
      {
        trees[4] = std::make_shared<PBBFMM_kelvin_3d_yz>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[4]->buildFMMTree();            
      }
      #pragma omp section
      {
        trees[5] = std::make_shared<PBBFMM_kelvin_3d_zz>
            (param.L, param.tree_level, param.interpolation_order,
             param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
        trees[5]->buildFMMTree();            
      }          
    }
  }
}

// ================ BBFMM3D =======================
class BBFMM_laplace_2d : public bbfmm3::H2_3D_Tree
{
 public:
  double reg_eps_;
  
  BBFMM_laplace_2d(double L, int level, int n, double epsilon, int use_chebyshev, double reg_eps)
      : bbfmm3::H2_3D_Tree(L, level, n, epsilon, use_chebyshev),
        reg_eps_(reg_eps)
  {
    std::cout << "[BBFMM_laplace2d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void setHomogen(std::string& kernelType, bbfmm3::doft *dof)
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "BBFMM_laplace_2d";
    dof->s = 1;
    dof->f = 1;
  }
  virtual void EvaluateKernel(bbfmm3::vector3 targetpos,
                              bbfmm3::vector3 sourcepos,
                              double *K,
                              bbfmm3::doft *dof)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r2 = dx*dx+dy*dy+dz*dz+reg_eps_*reg_eps_;
    *K = -R_4PI*log(r2);
  }
};

class BBFMM_laplace_3d : public bbfmm3::H2_3D_Tree
{
 public:
  double reg_eps_;
  
  BBFMM_laplace_3d(double L, int level, int n, double epsilon, int use_chebyshev, double reg_eps)
      : bbfmm3::H2_3D_Tree(L, level, n, epsilon, use_chebyshev),
        reg_eps_(reg_eps)
  {
    std::cout << "[BBFMM_laplace3d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void setHomogen(std::string& kernelType, bbfmm3::doft *dof)
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "BBFMM_laplace_3d";
    dof->s = 1;
    dof->f = 1;
  }
  virtual void EvaluateKernel(bbfmm3::vector3 targetpos,
                              bbfmm3::vector3 sourcepos,
                              double *K,
                              bbfmm3::doft *dof)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r2 = dx*dx+dy*dy+dz*dz+reg_eps_*reg_eps_;   
    *K = R_4PI/sqrt(r2);
  }  
};

class BBFMM_bilaplap_3d : public bbfmm3::H2_3D_Tree
{
 public:
  double reg_eps_;
  
  BBFMM_bilaplap_3d(double L, int level, int n, double epsilon, int use_chebyshev, double reg_eps)
      : bbfmm3::H2_3D_Tree(L, level, n, epsilon, use_chebyshev),
        reg_eps_(reg_eps)
  {
    std::cout << "[BBFMM_bilaplap3d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void setHomogen(std::string& kernelType, bbfmm3::doft *dof)
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "BBFMM_bilaplap_3d";
    dof->s = 1;
    dof->f = 1;
  }
  virtual void EvaluateKernel(bbfmm3::vector3 targetpos,
                              bbfmm3::vector3 sourcepos,
                              double *K,
                              bbfmm3::doft *dof)
  {
    double dx = sourcepos.x - targetpos.x;
    double dy = sourcepos.y - targetpos.y;
    double dz = sourcepos.z - targetpos.z;
    double r  = sqrt(dx*dx+dy*dy+dz*dz);
    *K = r < 1e-10 ? R_4PI/reg_eps_ : R_4PI*(1-exp(-r/reg_eps_))/r;
  }  
};

class BBFMM_kelvin_3d : public bbfmm3::H2_3D_Tree
{
 public:
  double reg_eps_, a_, b_;
  
  BBFMM_kelvin_3d(double L, int level, int n, double epsilon, int use_chebyshev, double a, double b, double reg_eps)
      : bbfmm3::H2_3D_Tree(L, level, n, epsilon, use_chebyshev),
        a_(a), b_(b), reg_eps_(reg_eps)
  {
    std::cout << "[BBFMM_kelvin3d] reg_eps_=" << reg_eps_ << std::endl;
  }
  virtual void setHomogen(std::string& kernelType, bbfmm3::doft *dof)
  {
    homogen  = 0;
    symmetry = 1;
    kernelType = "BBFMM_kelvin_3d";
    dof->s = 3;
    dof->f = 3;
  }
  virtual void EvaluateKernel(bbfmm3::vector3 targetpos,
                              bbfmm3::vector3 sourcepos,
                              double *K,
                              bbfmm3::doft *dof)
  {
    Eigen::Vector3d r(
        sourcepos.x - targetpos.x,
        sourcepos.y - targetpos.y,
        sourcepos.z - targetpos.z);
    const double reg_r1 = sqrt(r.squaredNorm()+reg_eps_*reg_eps_);
    const double reg_r3 = reg_r1*reg_r1*reg_r1;
    Eigen::Map<Eigen::Matrix3d>(K, 3, 3)
        = (a_-b_)/reg_r1*Eigen::Matrix3d::Identity()
        +b_/reg_r3*r*r.transpose();
  }  
};

void create_fmm_tree(std::shared_ptr<bbfmm3::H2_3D_Tree> &tree,
                     const KERNEL_TYPE kernel,
                     const FMM_parameters &param)
{
  switch ( kernel )
  {
    case LAPLACE_2D:
      tree = std::make_shared<BBFMM_laplace_2d>
          (param.L, param.tree_level, param.interpolation_order,
           param.eps, param.use_chebyshev, param.reg_eps);
      break;
    case LAPLACE_3D:
      tree = std::make_shared<BBFMM_laplace_3d>
          (param.L, param.tree_level, param.interpolation_order,
           param.eps, param.use_chebyshev, param.reg_eps);
      break;
    case BILAP_REG_3D:
      tree = std::make_shared<BBFMM_bilaplap_3d>
          (param.L, param.tree_level, param.interpolation_order,
           param.eps, param.use_chebyshev, param.reg_eps);
      break;
    case KELVIN_3D:
      tree = std::make_shared<BBFMM_kelvin_3d>
          (param.L, param.tree_level, param.interpolation_order,
           param.eps, param.use_chebyshev, param.a, param.b, param.reg_eps);
      break;
    default:
      ASSERT(0);
  }
}

// ------------------- register kernels to PDE ------------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename real_t>
struct gf_summary;

template <typename scalar_t, typename real_t>
struct gf_summary<PDE_TYPE::POISSON, scalar_t, real_t>
{
  __device__ __host__ __forceinline__
  static void run(const int      id,
                  const real_t   *x,
                  const real_t   *y,
                  const real_t   *n_x,
                  const real_t   *n_y,
                  const real_t   *p,
                  scalar_t       *G)
  {
    switch ( id ) {
      case KERNEL_TYPE::LAPLACE_2D:
        laplace_2d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::LAPLACE_3D:
        laplace_3d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::SCR_POISSON_2D:
        screen_poisson_2d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::BILAP_REG_3D:
        bilap_reg_3d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::LAPLACE_NM_2D:
        laplace_nm_2d(x, y, n_x, n_y, p, G);
        break;
      default:
        printf("# unsupported kernel!\n");        
    }
  }

  // FOR LEAST-SQUARE
  __device__ __forceinline__
  static void run(const int i, const int j,
                  const int n_bnd,
                  const scalar_t *K_mat,
                  scalar_t *G)
  {
    // G_{ij} = K(:,i)^T K(:,j)
    *G = thrust::inner_product(
        thrust::device,
        K_mat+i*n_bnd,
        K_mat+(i+1)*n_bnd,
        K_mat+j*n_bnd,
        scalar_t(0),
        thrust::plus<scalar_t>(),
        thrust::multiplies<scalar_t>());
  }
};

template <typename scalar_t, typename real_t>
struct gf_summary<PDE_TYPE::KELVIN, scalar_t, real_t>
{
  __device__ __host__ __forceinline__
  static void run(const int      id,
                  const real_t   *x,
                  const real_t   *y,
                  const real_t   *n_x,
                  const real_t   *n_y,
                  const real_t   *p,
                  scalar_t       *G)
  {
    if ( id == static_cast<int>(KERNEL_TYPE::KELVIN_3D) ) {
      return kelvin_3d(x, y, n_x, n_y, p, G);
    }

    printf("# unsupported kernel!\n");
    return;
  }

  // FOR LEAST-SQUARE
  __device__ __forceinline__
  static void run(const int i, const int j, const int n_bnd,
                  const scalar_t *K_mat,
                  scalar_t *G)
  {
    // G_{3i:3i+3, 3j:3j+3} = K(:,3i:3i+3)^T K(:,3j:3j+3)
    #pragma unroll 9
    for (int iter = 0; iter < 9; ++iter) {
      const int r = iter%3, c = iter/3;
      *(G+iter) = thrust::inner_product(
          thrust::device,
          K_mat+(3*i+r)*3*n_bnd,
          K_mat+(3*i+r+1)*3*n_bnd,
          K_mat+(3*j+c)*3*n_bnd,
          scalar_t(0),
          thrust::plus<scalar_t>(),
          thrust::multiplies<scalar_t>());      
    }
  }
};

template <typename scalar_t, typename real_t>
struct gf_summary<PDE_TYPE::HELMHOLTZ, scalar_t, real_t>
{
  __device__ __host__ __forceinline__
  static void run(const int      id,
                  const real_t   *x,
                  const real_t   *y,
                  const real_t   *n_x,
                  const real_t   *n_y,
                  const real_t   *p,
                  scalar_t       *G)
  {
    switch ( id ) {
      case KERNEL_TYPE::HELMHOLTZ_2D:
        helmholtz_2d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::HELMHOLTZ_NM_2D:
        helmholtz_nm_2d(x, y, n_x, n_y, p, G);
        break;
      case KERNEL_TYPE::HELMHOLTZ_3D:
        helmholtz_3d(x, y, n_x, n_y, p, G);
        break;
      default:
        printf("# unsupported kernel!\n");
    }
  }

  // FOR LEAST-SQUARE
  __device__ __forceinline__
  static void run(const int i, const int j, const int n_bnd,
                  const scalar_t *K_mat,
                  scalar_t *G)
  {
    // G_{ij} = K(:,i)^H K(:,j)
    *G = thrust::inner_product(
        thrust::device,
        K_mat+i*n_bnd,
        K_mat+(i+1)*n_bnd,
        K_mat+j*n_bnd,
        scalar_t(0),
        thrust::plus<scalar_t>(),
        [](const scalar_t &a, const scalar_t &b) {
          return thrust::conj(a)*b;
        });
  }
};

}
#endif
