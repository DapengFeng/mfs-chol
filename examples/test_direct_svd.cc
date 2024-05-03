//#define EIGEN_USE_LAPACKE
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <imgui.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/timer.h"
#include "src/io.h"
#include "src/gputimer.h"
#include "src/ptree.h"

using namespace std;
using namespace Eigen;
using namespace klchol;

enum PLOT_OPTION
{
  GREEN_FUNC,
  SPARSITY,
  LHSxX,
  RHS,
  SCALE
};

enum SOLVER_TYPE
{
  KLCHOL,
  LDLT
};

static int SAVE_COUNT = 1;

template <class Cont1, class Cont2>
static int save_mesh_and_data(const std::string &dir,
                              const RmMatF_t &V,
                              const RmMatI_t &F,
                              const Cont1 &data,
                              const Cont2 &param)
{
  char outf[256];

  // write mesh
  sprintf(outf, "%s/mesh-%03d.obj", dir.c_str(), SAVE_COUNT);
  igl::writeOBJ(outf, V, F);

  // write solutions
  sprintf(outf, "%s/solution-%03d.dat", dir.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)data.data(), data.size()*sizeof(data[0]));
    ofs.close();
  }

  // write parameters
  sprintf(outf, "%s/param-%03d.dat", dir.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)param.data(), param.size()*sizeof(param[0]));
    ofs.close();
  }
  
  return 0;
}

static void merge_meshes(const RmMatF_t &V1, const RmMatI_t &F1,
                         const RmMatF_t &V2, const RmMatI_t &F2,
                         RmMatF_t &V, RmMatI_t &F)
{
  const size_t total_v = V1.rows()+V2.rows();
  V.resize(total_v, V1.cols());
  V.topRows(V1.rows()) = V1;
  V.bottomRows(V2.rows()) = V2;

  const size_t total_f = F1.rows()+F2.rows();
  F.resize(total_f, F1.cols());
  F.topRows(F1.rows()) = F1;
  F.bottomRows(F2.rows()) = F2.array()+V1.rows();
}

// thrust based complex number
typedef thrust::complex<double> complex_t;
typedef Eigen::SparseMatrix<thrust::complex<double>> complex_spmat_t;
typedef Eigen::Matrix<thrust::complex<double>, -1, 1> complex_vec_t;
typedef Eigen::Matrix<thrust::complex<double>, -1, -1> complex_mat_t;

struct input_flow_t
{
  Vector3d p_;   // center
  double   k_;   // wavenumber

  input_flow_t(const double k)
      : k_(k)
  {    
  }
  input_flow_t(const Vector3d &p, const double k)
      : p_(p), k_(k)
  {    
  }
  complex_vec_t get_value(const size_t npts, const double *x) {
    complex_vec_t rtn(npts);
    #pragma omp parallel for
    for (size_t i = 0; i < npts; ++i) {
      rtn[i] = complex_t(cos(k_*x[3*i+0]), sin(k_*x[3*i+0]));
    }
    return rtn;
  }
  complex_vec_t get_dervn(const size_t npts, const double *x, const double *n) {
    complex_vec_t rtn(npts);
    #pragma omp parallel for
    for (size_t i = 0; i < npts; ++i) {
      rtn[i] = complex_t(-k_*sin(k_*x[3*i+0])*n[3*i+0], k_*cos(k_*x[3*i+0])*n[3*i+0]);
    }
    return rtn;
  }
};

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  RmMatF_t Vb, Vs, Vq;    // boundary, source, targets and merged mesh
  RmMatI_t Fb, Fs, Fq;    // faces

  const string SOURCE_MESH   = pt.get<string>("source_mesh.value");
  const string TARGET_MESH   = pt.get<string>("target_mesh.value");
  const string BOUNDARY_MESH = pt.get<string>("boundary_mesh.value");
  const string OUTDIR        = pt.get<string>("outdir.value");  

  // read meshes
  igl::readOFF(SOURCE_MESH,   Vs, Fs);  
  igl::readOFF(BOUNDARY_MESH, Vb, Fb);
  igl::readOFF(TARGET_MESH,   Vq, Fq);

  // normalize meshes
  {
    Eigen::RowVector3d C = Vq.colwise().sum()/Vq.rows();
    Vs.rowwise() -= C;
    Vb.rowwise() -= C;
    Vq.rowwise() -= C;
    
    Eigen::RowVector3d BBOX = Vq.colwise().maxCoeff()-Vq.colwise().minCoeff();
    const double max_span = BBOX.maxCoeff();
    Vs /= max_span;
    Vb /= max_span;    
    Vq /= max_span;
  }

  // write boundary mesh
  igl::writeOFF(string(OUTDIR+"/boundary_mesh.off"), Vb, Fb);
  
  const double EPS = pt.get<double>("eps.value");
  ASSERT(EPS < 1.0);

  spdlog::info("source mesh={}", Vs.rows());
  spdlog::info("target mesh={}", Vq.rows());
  spdlog::info("boundary mesh={}", Vb.rows());

  GpuTimer gpu_timer;
  
  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(Vs);
  gpu_timer.start();
  {
    fps.compute('F');
  }
  gpu_timer.stop();
  double FPS_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(fps)={0:.3f}", FPS_TIME);
  fps.reorder_geometry(Vs, Fs);
  fps.debug();

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu_config, menu_stat;  
  plugin.widgets.push_back(&menu_config);
  plugin.widgets.push_back(&menu_stat);  

  const Eigen::RowVector3d orange(1.0, 0.7, 0.2);  

  // parameters
  float  RHO = pt.get<float>("rho.value");  
  double KL_ERROR = 0, PCG_ERROR = 0, PCG_ERROR_NO_PREC = 0;
  float MAX_NNZ_J = 0,   MEAN_NNZ_J = 0;
  float MAX_SUPER_J = 0, MEAN_SUPER_J = 0;
  int   VERT_NUM = Vs.rows();
  PLOT_OPTION plot = LHSxX;
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::HELMHOLTZ_2D;
  SOLVER_TYPE SOL_TYPE = SOLVER_TYPE::KLCHOL;
  float EDGE_LENGTH = 0;
  float PAT_TIME = 0, CPY_TIME = 0, FAC_TIME = 0, SLV_TIME = 0, EVL_TIME = 0,
      PCG_TIME = 0, PCG_TIME_NO_PREC = 0;
  float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
  int   NUM_SUPERNODES = 0;
  int   MAX_SUPERNODE_SIZE = pt.get<int>("max_supernode_size.value");
  float WAVE_NUM = 0;
  int   PCG_MAXITS = pt.get<int>("pcg_iters.value");
  int   NUM_SEC = pt.get<int>("num_sec.value");
  float HELM_K =  pt.get<float>("helm_k.value");
  bool  USE_SVD =  pt.get<bool>("use_svd.value", false);

  // edge lengths
  VectorXd Le;
  igl::edge_lengths(Vs, Fs, Le);  
  EDGE_LENGTH = Le.sum()/Le.size();
  
  menu_config.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_config.menu_scaling(), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Config", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::InputDouble("EPS", &EPS, 0.001, 0.01);
    ImGui::SliderFloat("RHO", &RHO, 1.0f, 20.0f);
    ImGui::Combo("kernal", reinterpret_cast<int*>(&KTYPE), KERNEL_LIST);
    ImGui::Combo("solver", reinterpret_cast<int*>(&SOL_TYPE), "klchol\0ldlt\0\0");
    ImGui::SliderFloat("freq(rhs)", &WAVE_NUM, 0.0f, 100.0f);
    ImGui::SliderFloat("helmholtz k", &HELM_K, 0.0f, 200.0f);
    ImGui::Combo("plot", reinterpret_cast<int*>(&plot), "GreenFunc\0Sparsity\0lhs*x\0rhs\0length_scale\0\0");
    ImGui::InputInt("max supernode size", &MAX_SUPERNODE_SIZE, 1, 10000);
    ImGui::SliderInt("PCG maxits", &PCG_MAXITS, 1.0, 100);
    ImGui::Checkbox("use_svd", &USE_SVD);
    ImGui::End();
  };

  menu_stat.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_stat.menu_scaling(), 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::BulletText("vert num=%d", VERT_NUM);
    ImGui::BulletText("edge length=%f", EDGE_LENGTH);    
    //    ImGui::BulletText("bbox size=(%f %f %f)", BBOX.x(), BBOX.y(), BBOX.z());
    ImGui::Text("---------------------------------------------");
    ImGui::BulletText("number of supernodes=%d", NUM_SUPERNODES);
    ImGui::BulletText("nnz(simpl_sparsity)=%.1f", SIMPL_NNZ);
    ImGui::BulletText("nnz(super_sparsity)=%.1f", SUPER_NNZ);
    ImGui::BulletText("size(THETA)=%.1f",         THETA_NNZ);
    ImGui::BulletText("max_col(super_sparsity)=%.1f", MAX_NNZ_J );
    ImGui::BulletText("ave_col(super_sparsity)=%.1f", MEAN_NNZ_J);    
    ImGui::BulletText("max(supernodes)=%.1f", MAX_SUPER_J );    
    ImGui::BulletText("ave(supernodes)=%.1f", MEAN_SUPER_J);
    ImGui::BulletText("number of sections=%d", NUM_SEC);
    ImGui::Text("---------------------------------------------");
    ImGui::BulletText("|Ax-b|/|b|=%lf", KL_ERROR);
    ImGui::BulletText("pcg error=%lf",  PCG_ERROR);
    ImGui::BulletText("non-pcg error=%lf", PCG_ERROR_NO_PREC);
    ImGui::Text("---------------------------------------------");    
    ImGui::BulletText("time(calc pattern)   %.2f ms", PAT_TIME);
    ImGui::BulletText("time(data copy)      %.2f ms", CPY_TIME);
    ImGui::BulletText("time(factorize)      %.2f ms", FAC_TIME);
    ImGui::BulletText("time(solve)          %.2f ms", SLV_TIME);
    ImGui::BulletText("time(evalKx)         %.2f ms", EVL_TIME);
    ImGui::BulletText("time(pcg)            %.2f ms", PCG_TIME);
    ImGui::BulletText("time(non-pcg)        %.2f ms", PCG_TIME_NO_PREC);
    ImGui::End();
  };
  
  // init the gpu cholesky solver
  const size_t DIM = 1;
  const size_t N = Vs.rows();
  const std::vector<size_t> GROUP{0, N};  
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::HELMHOLTZ>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::HELMHOLTZ>(N, DIM, NUM_SEC));  
  super_solver->set_source_points(Vs.rows(), Vs.data());
  super_solver->set_target_points(Vq.rows(), Vq.data());

  // set sources and boundary points to assemble covariance
  super_solver->ls_cov_->set_source_pts(Vs.rows(), Vs.data());
  super_solver->ls_cov_->set_bound_pts(Vb.rows(), Vb.data());

  complex_spmat_t PATT, SUP_PATT;
  complex_vec_t y(N); y.setZero();
  VectorXd PARAM = VectorXd::Zero(8), plot_data(Vq.rows());
  complex_vec_t potent(Vq.rows()), rhs(Vs.rows());

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

  Eigen::MatrixXcd G_LHS;
  Eigen::VectorXcd G_rhs_bnd, G_y;
  
  VectorXi sup_ptr, sup_ind, sup_parent;  
  const auto &update
      = [&]()
        {
          PARAM[0] = EPS;
          PARAM[1] = HELM_K;
          PARAM[2] = RHO;
          PARAM[3] = KTYPE;
          PARAM[6] = 1.0;
          PARAM[7] = 1.0;
          gpu_timer.start();
          {
            super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
            super_solver->ls_cov_->build_cov_mat_LS(KTYPE, PARAM.data(), PARAM.size());
          }
          gpu_timer.stop();
          double TIME_BUILD_COV = gpu_timer.elapsed()/1000;
          spdlog::info("build covmat time={0:.3f}", TIME_BUILD_COV);

          gpu_timer.start();
          {
            fps.simpl_sparsity(RHO, DIM, PATT);
            fps.aggregate(DIM, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
            fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);
          }
          gpu_timer.stop();
          PAT_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(PATT)={0:.3f}", PAT_TIME);

          NUM_SUPERNODES = sup_ptr.size()-1;
          SIMPL_NNZ = PATT.nonZeros();
          SUPER_NNZ = SUP_PATT.nonZeros();

          gpu_timer.start();
          {
            super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
            super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
          }
          gpu_timer.stop();
          CPY_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(cpy)={0:.3f}", CPY_TIME);

          THETA_NNZ = super_solver->theta_nnz();          
          analyse_nnz(SUP_PATT.cols(), SUP_PATT.outerIndexPtr(), MAX_NNZ_J, MEAN_NNZ_J);
          analyse_nnz(sup_ptr.size()-1, &sup_ptr[0], MAX_SUPER_J, MEAN_SUPER_J);

          gpu_timer.start();
          {
            super_solver->assemble();
          }
          gpu_timer.stop();          
          double ASM_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(asm.)={0:.3f}", ASM_TIME);

          gpu_timer.start();
          {
            super_solver->factorize();
          }
          gpu_timer.stop();
          FAC_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(fac.)={0:.3f}", FAC_TIME);
          spdlog::info("TIME(compute)={0:.3f}", FAC_TIME+ASM_TIME);
          spdlog::info("TIME(precompute)={0:.3f}", FPS_TIME+PAT_TIME+CPY_TIME);

          // rhs_bnd
          input_flow_t inflow(HELM_K);
          complex_vec_t rhs_bnd = -inflow.get_value(Vb.rows(), Vb.data());
          // rhs_source = A^T*rhs_bnd
          super_solver->ls_cov_->build_cov_rhs_LS(
              rhs_bnd.data(), rhs_bnd.size(), rhs.data(), rhs.size());

          // PCG
          gpu_timer.start();
          {
            y.setZero();
            auto ret = super_solver->pcg(
                rhs.data(), y.data(), true, PCG_MAXITS, 1e-3);
            PCG_ERROR = ret.second;
          }
          gpu_timer.stop();
          PCG_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(pcg)={0:.3f}", PCG_TIME);
          spdlog::info("pcg error={0:.6f}", PCG_ERROR);
          spdlog::info("TIME(total)={0:.3f}", FPS_TIME+PAT_TIME+CPY_TIME+FAC_TIME+PCG_TIME);

          // // pcg without preconditioner
          // gpu_timer.start();          
          // {
          //   y.setZero();
          //   auto pcg_ret = super_solver->pcg(rhs.data(), y.data(), false, 100, PCG_ERROR);
          //   PCG_ERROR_NO_PREC = pcg_ret.second;
          //   spdlog::info("non-pcg iters={}", pcg_ret.first);
          //   spdlog::info("non-pcg error={0:.6f}", PCG_ERROR_NO_PREC);
          // }
          // gpu_timer.stop();
          // spdlog::info("non-pcg time={0:.3f}", gpu_timer.elapsed());

          if ( USE_SVD ) {
            // LHS
            G_LHS.resize(super_solver->ls_cov_->rows(), super_solver->ls_cov_->cols());
            super_solver->ls_cov_->K(reinterpret_cast<complex_t*>(G_LHS.data()));

            // rhs
            G_rhs_bnd.resize(rhs_bnd.size());
            memcpy(reinterpret_cast<complex_t*>(G_rhs_bnd.data()), rhs_bnd.data(), rhs_bnd.size()*sizeof(complex_t));
            
            cout << "start svd..." << endl;
            gpu_timer.start();
            Eigen::BDCSVD<decltype(G_LHS)> svd(G_LHS, ComputeFullU | ComputeFullV);
            gpu_timer.stop();
            const double TIME_SVD_COMP = gpu_timer.elapsed()/1000;
            spdlog::info("TIME(svdcomp)={0:.3f}", TIME_SVD_COMP);

            gpu_timer.start();
            {
              G_y = svd.solve(G_rhs_bnd);
            }
            gpu_timer.stop();
            const double TIME_SVD_SLV = gpu_timer.elapsed()/1000;
            spdlog::info("TIME(svdsolve)={0:.3f}", TIME_SVD_SLV);

            VectorXcd temp_y(y.size());
            memcpy(reinterpret_cast<complex_t*>(temp_y.data()), y.data(), y.size()*sizeof(y[0]));
            cout << "error=" << (G_LHS.adjoint()*(G_LHS*(G_y-temp_y))).norm()/(G_LHS.adjoint()*G_rhs_bnd).norm() << endl;
          }
          
          // cuda-based accurate extrapolation
          gpu_timer.start();
          {
            potent.setZero();
            super_solver->predict(
                1, y.data(), y.size(),
                potent.data(), potent.size());
            // add input flow
            potent += inflow.get_value(Vq.rows(), Vq.data());
          }
          gpu_timer.stop();
          const double PRED_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(CUDA)={0:.3f}", PRED_TIME);
          
          #pragma omp parallel for
          for (size_t j = 0; j < Vq.rows(); ++j) {
            plot_data[j] = potent[j].real();
            // plot_data[j] = sqrt(potent[j].real()*potent[j].real()+potent[j].imag()*potent[j].imag());
          }
          std::cout << "plot max=" << plot_data.maxCoeff() << endl;
          viewer.data().set_data(plot_data, igl::COLOR_MAP_TYPE_PARULA);
          viewer.data().add_points(Vb, orange);
        };
      
  viewer.callback_mouse_move =
      [&](igl::opengl::glfw::Viewer& viewer, int, int) -> bool {
        return false;
      };

  viewer.callback_mouse_up =
      [&](igl::opengl::glfw::Viewer& viewer, int, int) -> bool {
        return false;
      };

  viewer.callback_key_pressed = 
      [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
      {
        switch(key) { 
          case 'u':
          case 'U':
            update();
            spdlog::info("save mesh and data {} times", SAVE_COUNT);
            save_mesh_and_data(OUTDIR, Vq, Fq, potent, PARAM);
            ++SAVE_COUNT;
            break;
          default:
            return false;
        }
        return true;
      };  

  viewer.data().set_mesh(Vq, Fq);
  viewer.data().show_lines = false;
  viewer.core().align_camera_center(Vq, Fq);
  viewer.launch();
  
  return 0;
}
