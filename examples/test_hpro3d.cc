#include <iostream>
#include <Eigen/Eigen>
#include <spdlog/spdlog.h>
#include <hpro.hh>
#include <random>

#include "src/types.h"
#include "src/preprocess.h"
#include "src/timer.h"
#include "src/gputimer.h"
#include "src/kl_chol.h"
#include "src/ptree.h"
#include "src/poisson_disk_sampling.h"

using namespace std;
using namespace Hpro;
using namespace Eigen;
using namespace klchol;

using value_t = double;
using real_t  = real_type_t<value_t>;

#define R_4PI 0.07957747154594767

class DiffuseProb : public TCoeffFn<value_t>
{
 public:
  using TCoeffFn< value_t >::eval;
  
  DiffuseProb(const RmMatF_t &Vq, const double eps)
      : V_(Vq), eps_(eps)
  {    
  }  
  virtual void eval(const std::vector<idx_t> &  rowidxs,
                    const std::vector<idx_t> &  colidxs,
                    value_t *                   matrix) const
  {
    const size_t n = rowidxs.size();
    const size_t m = colidxs.size();
    
    const value_t eps2 = eps_*eps_;
 
    for ( size_t  j = 0; j < m; ++j ) {
      const int  idx1 = colidxs[j];
            
      for ( size_t  i = 0; i < n; ++i ) {
        const int  idx0 = rowidxs[i];

        const value_t r2 = (V_.row(idx1)-V_.row(idx0)).squaredNorm()+eps2;
        matrix[j*n + i] = R_4PI/sqrt(r2);
      }
    }
  }
  virtual matform_t  matrix_format  () const { return symmetric; } 
  virtual bool       is_complex     () const { return false; }

 private:
  RmMatF_t V_;
  double eps_;
};

int main (int argc, char **argv) 
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);

  const size_t guessN = pt.get<size_t>("guess_n.value");
  const int DIM = 1;  
  const double EPS = pt.get<double>("eps.value");
  const klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_3D;
  const int MAX_SUPERNODE_SIZE = INT_MAX;

  // poisson disk sampling
  auto kRadius = cbrt(1.5/M_PI/guessN);
  auto kXMin = std::array<double, 3>{{-0.5, -0.5, -0.5}};
  auto kXMax = std::array<double, 3>{{+0.5, +0.5, +0.5}};
  auto samples = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
  const size_t N = samples.size();
  RmMatF_t V1 = RmMatF_t::Zero(N, 3);
  for (size_t i = 0; i < N; ++i) {
    V1(i, 0) = samples[i][0];
    V1(i, 1) = samples[i][1];
    V1(i, 2) = samples[i][2];
  }
  spdlog::info("N={}", N);

  // fmm 
  FMM_parameters fmm_config;
  {
    fmm_config.L                   = 1.0;
    fmm_config.tree_level          = V1.rows() > 180000 ? 6 : 5; // emperical
    fmm_config.interpolation_order = pt.get<int>("fmm_order.value", 4);
    fmm_config.eps                 = pt.get<double>("fmm_eps.value", 1e-6);
    fmm_config.use_chebyshev       = 1;
    fmm_config.reg_eps             = EPS;
    fmm_config.K                   = 0;
  }  
  
  // PBBFMM data storage
  std::vector<vector3> sources(N);
  for (size_t i = 0; i < N; ++i) {
    sources[i].x = V1(i, 0);
    sources[i].y = V1(i, 1);
    sources[i].z = V1(i, 2);      
  }  

  GpuTimer timer;
  double PREC_TIME = 0;
  
  INIT();
  CFG::set_verbosity(3);

  // bind coordinates
  std::vector<double*> vertices(N); 
  for (size_t i = 0; i < N; i++) {
    vertices[i] = &V1(i, 0);
  } 
  auto coord = make_unique<TCoordinate>(vertices, 3);

  timer.start();
  TAutoBSPPartStrat part_strat;
  TBSPCTBuilder ct_builder( & part_strat, 20 );
  auto ct = ct_builder.build( coord.get() );
  timer.stop();
  const double TIME_A = timer.elapsed()/1000;
  spdlog::info("time(ct_build)={0:.3f}", TIME_A);

  timer.start();
  TStdGeomAdmCond adm_cond( 2.0 );
  TBCBuilder bct_builder;
  auto bct = bct_builder.build( ct.get(), ct.get(), &adm_cond );
  timer.stop();
  const double TIME_B = timer.elapsed()/1000;
  spdlog::info("time(bct_build)={0:.3f}", TIME_B);
  
  // compute matrix coefficients
  timer.start();
  DiffuseProb prob(V1, EPS);
  TPermCoeffFn<value_t> coefffn(&prob, ct->perm_i2e(), ct->perm_i2e());
  TACAPlus<value_t> aca(&coefffn);
  TDenseMatBuilder<value_t> h_builder(&coefffn, &aca);
  TTruncAcc acc(pt.get<double>("acc_tol.value"), 0.0);
  timer.stop();
  const double TIME_C = timer.elapsed()/1000;
  spdlog::info("time(coefficient)={0:.3f}", TIME_C);

  // assemble LHS A
  timer.start();
  auto A = h_builder.build( bct.get(), acc);
  timer.stop();
  const double TIME_D = timer.elapsed()/1000;
  spdlog::info("time(build A)={0:.3f}", TIME_D);

  // random 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
    
  // assemble RHS
  auto b = A->row_vector();
  VectorXd RHS(N);
  for (size_t i = 0; i < N; ++i) {
    RHS(i) = dis(gen) > 0 ? 1.0 : -1.0;
    b->set_entry(i, RHS(i));
  }
  timer.start();
  {
    ct->perm_e2i()->permute(b.get());
  }
  timer.stop();
  const double TIME_E = timer.elapsed()/1000;
  spdlog::info("b norm={0:.6f}", RHS.norm());        

  VectorXd X = VectorXd::Zero(N);
  TSolverInfo solve_info;
  const int num_iters = pt.get<int>("num_iters.value");
  double TIME_ITER = 0, TIME_F = 0;

  // H-LU
  timer.start();
  auto B = A->copy();
  auto A_inv = factorise_inv(B.get(), acc);
  timer.stop();
  TIME_F = timer.elapsed()/1000;
  spdlog::info("time(LU)={0:.3f}", TIME_F);

  TStopCriterion sstop(num_iters, 0, 0);
  TCG solver(sstop);    

  // pcg
  auto x = A->col_vector(); 
  timer.start();
  {
    solver.solve(A.get(), x.get(), b.get(), A_inv.get(), &solve_info);
  }
  timer.stop();
  TIME_ITER = timer.elapsed()/1000;    
  ct->perm_i2e()->permute(x.get());
  for (size_t i = 0; i < N; ++i) {
    X(i) = x->entry(i);
  }

  // test accuracy
  VectorXd Kx = VectorXd::Zero(N);
  std::vector<std::shared_ptr<H2_3D_Tree>> fmm_tree;
  create_fmm_tree(fmm_tree, KTYPE, fmm_config);
  H2_3D_Compute<H2_3D_Tree> eval(*fmm_tree[0], sources, sources, X.data(), 1, Kx.data());
  const double RET_ERROR = (Kx-RHS).squaredNorm()/RHS.squaredNorm();
  spdlog::info("%% {0:d},{1:.3f},{2:e}", num_iters, TIME_ITER, RET_ERROR);
  spdlog::info("precompute time={0:.3f}", TIME_A+TIME_B+TIME_C+TIME_D+TIME_E+TIME_F);

  if ( pt.get<bool>("brute_force_eval.value", false) ) {
    // test accuracy: brute force
    VectorXd Resd = VectorXd::Zero(N);
    #pragma omp parallel for
    for (size_t i = 0; i < V1.rows(); ++i) {
      for (size_t j = 0; j < V1.rows(); ++j) {
        const double r2 = (V1.row(i)-V1.row(j)).squaredNorm()+EPS*EPS;
        Resd[i] += R_4PI/sqrt(r2)*X[j];
      }
      Resd[i] = Resd[i]-RHS[i];
    }
    spdlog::info("real residual={0:e}", Resd.squaredNorm()/RHS.squaredNorm());
  }
  
  DONE();
  return 0;
}
