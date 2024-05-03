#include <igl/readOFF.h>
#include <igl/per_vertex_normals.h>
#include <iostream>
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#include <spdlog/spdlog.h>
#include <random>

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/gputimer.h"
#include "src/io.h"
#include "src/ptree.h"
#include "src/poisson_disk_sampling.h"

using namespace std;
using namespace Eigen;
using namespace klchol;

template <typename FLOAT>
static void write_points(const string &file, const FLOAT *node, size_t node_num)
{
  ofstream os(file);
  os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

  os<< "POINTS " << node_num << " float\n";
  for(size_t i = 0; i < node_num; ++i)
    os << node[i*3+0] << " " << node[i*3+1] << " " << node[i*3+2] << "\n";

  auto points_num = node_num;
  os << "CELLS " << points_num << " " << points_num*2 << "\n";
  for(size_t i = 0; i < points_num; ++i)
    os << 1 << " " << i << "\n";

  os << "CELL_TYPES " << points_num << "\n";
  for(size_t i = 0; i < points_num; ++i)
    os << 1 << "\n";
}

template <class Cont>
static void write_residual(const string &filename, const Cont &resd,
                           const double dt, const double time_before_solve)
{
  ofstream ofs(filename);
  ofs << "iter, time, resd" << endl;
  ofs << "0, 0, 1.0" << endl;
  for (size_t i = 1; i <= resd.size(); ++i) {
    ofs << i << "," << time_before_solve+(i-1)*dt << "," << resd[i-1] << endl;
  }
  ofs.close();  
}

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);

  const string outdir = pt.get<string>("outdir.value");

  const size_t guessN = pt.get<size_t>("guess_n.value");
  const float RHO = pt.get<float>("rho.value");
  const int DIM = 1;  
  const float EPS = pt.get<double>("eps.value");
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
  spdlog::info("RHO={0:.2f}", RHO);

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

  GpuTimer timer;
  
  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(V1);
  timer.start();
  {
    fps.compute('F');
    fps.reorder_geometry(V1);
  }
  timer.stop();
  const double FPS_TIME = timer.elapsed()/1000;
  spdlog::info("TIME(FPS)={0:.3f} s", FPS_TIME);

  // fmm data format
  std::vector<vector3> sources(V1.rows());
  for (size_t i = 0; i < sources.size(); ++i) {
    sources[i].x = V1(i, 0);
    sources[i].y = V1(i, 1);
    sources[i].z = V1(i, 2);
  }
  
  // init the gpu cholesky solver
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON>(N, DIM, 1));
  super_solver->set_source_points(N, V1.data());  

  VectorXd param(8);  
  param[0] = EPS;
  super_solver->set_kernel(KTYPE, param.data(), param.size());  

  // sparsity pattern
  VectorXi sup_ptr, sup_ind, sup_parent;
  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  const std::vector<size_t> GROUP{0, N};
  timer.start();
  {
    fps.simpl_sparsity(RHO, DIM, PATT);
    fps.aggregate(DIM, GROUP, PATT, 1.3, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
    fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);
  }
  timer.stop();
  const float PAT_TIME = timer.elapsed()/1000;
  spdlog::info("TIME(PATT): {0:.3f}", PAT_TIME);

  const int NUM_SUPERNODES = sup_ptr.size()-1;
  const size_t SIMPL_NNZ = PATT.nonZeros();
  const size_t SUPER_NNZ = SUP_PATT.nonZeros();
  spdlog::info("num supernodes={}", NUM_SUPERNODES);
  spdlog::info("simpl sparsity nnz={}", SIMPL_NNZ);
  spdlog::info("super sparsity nnz={}", SUPER_NNZ);

  timer.start();
  {
    super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
    super_solver->set_sppatt(N, SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
  }
  timer.stop();
  const float CPY_TIME = timer.elapsed()/1000;
  spdlog::info("TIME(copy): {0:.3f}", CPY_TIME);

  timer.start();
  {
    super_solver->compute();
  }   
  timer.stop();
  const float FAC_TIME = timer.elapsed()/1000;
  spdlog::info("TIME(fac): {0:.3f}", FAC_TIME);

  spdlog::info("TIME(all_precomp)={0:.3f}", FAC_TIME+CPY_TIME+PAT_TIME+FPS_TIME);

  // random number
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  
  const int pcg_iters = pt.get<int>("pcg_iters.value");
  VectorXd b(N), x = VectorXd::Zero(N);
  for (size_t i = 0; i < N; ++i) {
    b(i) = dis(gen) > 0 ? 1.0 : -1.0;
  }
  spdlog::info("b norm={0:.6f}", b.norm());

  vector<double> pcg_resd;
  timer.start();
  {
    auto ret = super_solver->pcg(b.data(), x.data(), true, pcg_iters, 1e-6, &fmm_config, &pcg_resd);
    spdlog::info("pcg ret={0:e}", ret.second);
  }
  timer.stop();
  const float SLV_TIME = timer.elapsed()/1000;  
  spdlog::info("TIME(slv): {0:.3f}", SLV_TIME);
  {
    char outfile[256];
    sprintf(outfile, "%s/ours-pcg-guess-%zu-with-precomp-%d.txt", outdir.c_str(), guessN, pcg_iters);
    const double TIME_BEFORE_SOLVE = FPS_TIME+PAT_TIME+CPY_TIME+FAC_TIME;
    write_residual(outfile, pcg_resd, SLV_TIME/(pcg_resd.size()-1), TIME_BEFORE_SOLVE);
  }

  // test accuracy
  VectorXd Kx = VectorXd::Zero(N);
  std::vector<std::shared_ptr<H2_3D_Tree>> trees;    
  klchol::create_fmm_tree(trees, KTYPE, fmm_config);
  H2_3D_Compute<H2_3D_Tree> compute(*trees[0], sources, sources, x.data(), 1, Kx.data());
  spdlog::info("error={0:e}", (Kx-b).squaredNorm()/b.squaredNorm());

  spdlog::info("done");
  return 0;
}
