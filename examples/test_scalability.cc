#include <igl/readOFF.h>
#include <iostream>
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#include <spdlog/spdlog.h>
#include <random>

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/io.h"
#include "src/ptree.h"
#include "src/gputimer.h"
#include "src/poisson_disk_sampling.h"

using namespace std;
using namespace Eigen;
using namespace klchol;

enum SOLVER_TYPE
{
  KLCHOL,
  LDLT
};

extern "C" {
  int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
}

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  size_t Ns = pt.get<size_t>("num_sources.value");
  //  const size_t Nq = pt.get<size_t>("num_targets.value");
  const string OUTDIR = pt.get<string>("outdir.value");

  auto kRadius = sqrt(2.0/M_PI/Ns);
  auto kXMin = std::array<double, 2>{{-0.5, -0.5}};
  auto kXMax = std::array<double, 2>{{+0.5, +0.5}};
  auto samples = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
  Ns = samples.size();
  RmMatF_t Vs(Ns, 3);
  for (size_t i = 0; i < Ns; ++i) {
    Vs(i, 0) = samples[i][0];
    Vs(i, 1) = samples[i][1];
    Vs(i, 2) = 0;
  }
  spdlog::info("radius={0:.6f}", kRadius);
  spdlog::info("Ns={}", Ns);
  //  spdlog::info("Ns={}, Nq={}", Ns, Nq);

  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<> dis(-0.5, 0.5);  
  // RmMatF_t Vq(Nq, 3);
  // for (size_t i = 0; i < Nq; ++i) {
  //   Vq(i, 0) = dis(gen); 
  //   Vq(i, 1) = dis(gen); 
  //   Vq(i, 2) = 0;
  // }
  // VectorXd rhs(Ns);
  // for (size_t i = 0; i < Ns; ++i) {
  //   rhs[i] = dis(gen);
  // }
  
  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(Vs);
  fps.compute('F');
  fps.reorder_geometry(Vs);
  fps.debug();
  
  // parameters
  double EPS = pt.get<double>("eps.value");
  float  RHO = pt.get<float>("rho.value");
  double KL_ERROR = 0, PCG_ERROR = 0, PCG_ERROR_NO_PREC = 0;
  float MAX_NNZ_J = 0,   MEAN_NNZ_J = 0;
  float MAX_SUPER_J = 0, MEAN_SUPER_J = 0;
  int   VERT_NUM = Vs.rows();
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_3D;
  float EDGE_LENGTH = 0;
  float PAT_TIME = 0, CPY_TIME = 0, FAC_TIME = 0, SLV_TIME = 0, EVL_TIME = 0,
      PCG_TIME = 0, PCG_TIME_NO_PREC = 0, PRED_TIME = 0, FMM_TIME = 0;
  float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
  int   NUM_SUPERNODES = 0;
  int   MAX_SUPERNODE_SIZE = pt.get<int>("max_supernode_size.value");
  float WAVE_NUM = 0;
  int   NUM_SEC = pt.get<int>("num_sec.value");
    
  // init the gpu cholesky solver
  const size_t DIM = 1;
  const size_t N = Vs.rows();
  const std::vector<size_t> GROUP{0, N};
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON>(N, DIM, NUM_SEC));
  super_solver->set_source_points(Vs.rows(), Vs.data());

  GpuTimer gpu_timer;

  VectorXd PARAM = VectorXd::Zero(8);  
  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  VectorXi sup_ptr, sup_ind, sup_parent;
  
  // FMM_parameters fmm_config;
  // {
  //   fmm_config.L                   = 1;
  //   fmm_config.tree_level          = 4;
  //   fmm_config.interpolation_order = 3;
  //   fmm_config.eps                 = 5e-5;
  //   fmm_config.use_chebyshev       = 1;
  //   fmm_config.reg_eps             = EPS;
  // }
  // cout << "\tFMM.L=" << fmm_config.L << endl;
 
  const auto &analyse_nnz
      = [](const size_t n, const int *ptr, float &max_nnz, float &mean_nnz) 
        {
          max_nnz = 0;
          mean_nnz = 0;
          for (size_t j = 0; j < n; ++j) {
            size_t curr_nnz = ptr[j+1]-ptr[j];
            if ( curr_nnz > max_nnz ) {
              max_nnz = curr_nnz;
            }
            mean_nnz += curr_nnz;
          }
          mean_nnz /= n;
        };

  PARAM[0] = EPS;
  PARAM[2] = RHO;
  PARAM[7] = KTYPE;
  super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
  spdlog::info("rho={0:.3f}, eps={1:.9f}", RHO, EPS);
            
  gpu_timer.start();
  {
    fps.simpl_sparsity(RHO, DIM, PATT);
#if 0
    {
      std::vector<Triplet<double>> trips;
      for (size_t j = 0; j < PATT.cols(); ++j) {
        for (size_t i = j; i < PATT.rows(); ++i) {
          trips.emplace_back(Triplet<double>(i, j, 1.0));
        }
      }
      PATT.setFromTriplets(trips.begin(), trips.end());
    }
#endif                                
    fps.aggregate(DIM, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
    fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);
  }
  gpu_timer.stop();
  PAT_TIME = gpu_timer.elapsed();
  spdlog::info("TIME(patt_build)={0:.3f}", PAT_TIME);

  NUM_SUPERNODES = sup_ptr.size()-1;
  SIMPL_NNZ = PATT.nonZeros();
  SUPER_NNZ = SUP_PATT.nonZeros();
  PARAM[6] = 1.0*SUPER_NNZ/SUP_PATT.size();

  gpu_timer.start();
  {
    super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
    super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
  }
  gpu_timer.stop();
  CPY_TIME = gpu_timer.elapsed();
  spdlog::info("TIME(patt_copy)={0:.3f}", CPY_TIME);

  THETA_NNZ = super_solver->theta_nnz();

  analyse_nnz(SUP_PATT.cols(), SUP_PATT.outerIndexPtr(), MAX_NNZ_J, MEAN_NNZ_J);
  analyse_nnz(sup_ptr.size()-1, &sup_ptr[0], MAX_SUPER_J, MEAN_SUPER_J);

  gpu_timer.start();
  {              
    super_solver->compute();
  }
  gpu_timer.stop();
  FAC_TIME = gpu_timer.elapsed();
  spdlog::info("TIME(compute)={0:d}, {1:.3f}", Ns, FAC_TIME/1000);
  spdlog::info("memory={0:d}, {1:.6f}", Ns, super_solver->memory());

  if ( pt.get<bool>("count_loss.value") == false ) {
    cout << "# does not count loss" << endl;
    return 0;
  }

  Eigen::SparseMatrix<double> L;
  super_solver->get_factor(L);
  spdlog::info("L info={} {} {}", L.rows(), L.cols(), L.nonZeros());

  spdlog::info("compute K");
  MatrixXd K(Vs.rows(), Vs.rows());
  #pragma omp parallel for
  for (size_t i = 0; i < K.rows(); ++i) {
    for (size_t j = 0; j < K.cols(); ++j) {      
      klchol::laplace_3d<double, double>(
          &Vs(i, 0), &Vs(j, 0), nullptr, nullptr, PARAM.data(), &K(i, j));
    }
  }

  // reverse permutation
  Eigen::PermutationMatrix<-1, -1> P(N);
  for (size_t i = 0; i < P.size(); ++i) {
    P.indices()[i] = N-1-i;
  }

  MatrixXd U(N, N);
  {
    MatrixXd PKP = P*K*P;
    char uplo = 'L';
    int n = N;
    int info;
    dpotrf_(&uplo, &n, PKP.data(), &n, &info);
    U = PKP.triangularView<Eigen::Lower>();
    U = (P*U*P).eval();
  }

  spdlog::info("compute kaporin for matrix {}x{}", N, N);
  const double invN = 1.0/N;
  const MatrixXd LKL = L.transpose()*K*L;
  const double LKL_trace = LKL.trace();
  const double LKL_det = LKL.determinant();
  const double kaporin = invN*LKL_trace/pow(LKL_det, invN);
  const double KL_div = -log(LKL_det)+LKL_trace-N;
  const double FRO_norm = (MatrixXd::Identity(N, N)-L.transpose()*U).squaredNorm();
  spdlog::info("LKL norm={0:.9f}", LKL.norm());
  spdlog::info("trace={0:.9f}", LKL_trace);
  spdlog::info("determinant={0:.9f}", LKL_det);  
  spdlog::info("kaporin {0:.3f}, {1:.9f}, {2:.9f}, {3:.9f}, {4:.9f}, {5:.9f}, {6:.9f}", RHO, kaporin, log2(kaporin), KL_div, log10(KL_div), FRO_norm, log10(FRO_norm));
  
  cout << "# done!" << endl;
  return 0;
}

  // compute the kernel matrix

  // VectorXd y = VectorXd::Zero(Vs.rows());  
  // gpu_timer.start();
  // {
  //   auto pcg_ret = super_solver->pcg(rhs.data(), y.data(), true, PCG_MAXITS, 1e-4, &fmm_config);
  //   PCG_ERROR = pcg_ret.second;
  // }
  // gpu_timer.stop();
  // PCG_TIME = gpu_timer.elapsed();
  // spdlog::info("TIME(pcg)={0:.3f}", PCG_TIME);
  // spdlog::info("pcg error={0:.6f}", PCG_ERROR);

  // gpu_timer.start();
  // {
  //   auto ret = super_solver->pcg(rhs.data(), y.data(), false, PCG_MAXITS, 1e-4, &fmm_config);
  //   spdlog::info("non-pcg error={0:.6f}", ret.second);
  // }
  // gpu_timer.stop();
  
  // // fmm extrapolation
  // VectorXd fmm_potent = VectorXd::Zero(Vq.rows());
  // gpu_timer.start();
  // {
  //   super_solver->predict_fmm(
  //       1, y.data(), y.size(),
  //       fmm_potent.data(), fmm_potent.size(),
  //       fmm_config);
  // }
  // gpu_timer.stop();
  // FMM_TIME = gpu_timer.elapsed();
  // spdlog::info("TIME(fmm_extrap)={0:.3f}", FMM_TIME);
  // {
    
  // }
