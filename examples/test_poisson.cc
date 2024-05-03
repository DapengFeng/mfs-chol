#include <igl/kelvinlets.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/boundary_loop.h>
#include <imgui.h>
#include <iostream>
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#include <spdlog/spdlog.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/timer.h"
#include "src/io.h"
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

template <class Vec>
static int save_mesh_and_data(const std::string &dir,
                              const string &mesh_name,
                              const RmMatF_t &V,
                              const RmMatI_t &F,
                              const string &data_name, 
                              const Vec &data,
                              const string &param_name,
                              const Vec &param)
{
  typedef typename Vec::Scalar Scalar;

  char outf[256];

  // write mesh
  sprintf(outf, "%s/%s-%03d.obj", dir.c_str(), mesh_name.c_str(), SAVE_COUNT);
  igl::writeOBJ(outf, V, F);

  // write solutions
  sprintf(outf, "%s/%s-%03d.dat", dir.c_str(), data_name.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)data.data(), data.size()*sizeof(Scalar));
    ofs.close();
  }

  // write parameters
  sprintf(outf, "%s/%s-%03d.dat", dir.c_str(), param_name.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)param.data(), param.size()*sizeof(Scalar));
    ofs.close();
  }
  
  return 0;
}

template <typename scalar_t, typename index_t>
struct dense_chol_fac;

template <typename Scalar>
struct conj_dot : public thrust::binary_function<Scalar, Scalar, Scalar>
{
  __host__
  Scalar operator()(const Scalar &a, const Scalar &b)
  {
    return thrust::conj(a)*b;
  }
};

template <typename index_t>
struct dense_chol_fac<thrust::complex<double>, index_t>
{
  typedef thrust::complex<double> Complex;

  __host__
  static void run(const index_t n, Complex * __restrict__ rA)
  {
    for (index_t j = 0; j < n; ++j) {
      for (index_t i = 0; i <= j; ++i) {
        Complex dot_sum = 0;
        dot_sum = thrust::inner_product(thrust::host, rA+i*n, rA+i*n+i, rA+j*n,
                                        Complex(0.0),
                                        thrust::plus<Complex>(),
                                        conj_dot<Complex>());
        rA[i+j*n] = ( i == j ) ? sqrt((rA[j*n+j]-dot_sum).real()) : (rA[i+n*j]-dot_sum)/rA[i+i*n];
      }
    }
  }
};

template <typename scalar_t, typename index_t>
__host__
static void upper_tri_solve(const index_t                nU,
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


int main(int argc, char *argv[])
{
#if 0
  typedef Eigen::Matrix<std::complex<double>, -1, -1> std_matc;
  typedef Eigen::Matrix<thrust::complex<double>, -1, -1> thrust_matc;
  typedef Eigen::VectorXcd std_vecc;

  srand(time(NULL));
  const size_t Nx = 10;
  std_matc A = std_matc::Random(Nx, Nx);
  A = (A.adjoint()*A).eval();  

  thrust_matc C;
  C = A.cast<thrust::complex<double>>();

  dense_chol_fac<typename thrust_matc::Scalar, int>::run(Nx, C.data());
  std_matc resU(C.rows(), C.cols());
  for (size_t i = 0; i < resU.rows(); ++i) {
    for (size_t j = 0; j < resU.cols(); ++j) {
      resU(i, j) = C(i, j);
    }
  }
  cout << resU << endl << endl;

  std_matc L = std_matc(A.llt().matrixL());  
  cout << L.adjoint() << endl;

  std_vecc rhs_c = std_vecc::Random(Nx);
  cout << L.adjoint().triangularView<Eigen::Upper>().solve(rhs_c) << endl << endl;

  upper_tri_solve(Nx, resU.data(), Nx, rhs_c.data());
  cout << rhs_c << endl;
    
  return 0;
#endif

  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt); 

  RmMatF_t V1, Vq;    // vertices and normals
  RmMatI_t F1, Fq;    // faces
  Eigen::RowVector3d BBOX;

  const string MESH   = pt.get<string>("mesh.value");
  const string OUTDIR = pt.get<string>("outdir.value");
  const string QUERY_MESH = pt.get<string>("query_mesh.value");

  // read the mesh
  igl::readOFF(MESH, V1, F1);
  igl::readOFF(QUERY_MESH, Vq, Fq);
  {
    // normalize mesh dy diameter
    const Eigen::RowVector3d C = V1.colwise().sum()/V1.rows();
    BBOX = V1.colwise().maxCoeff()-V1.colwise().minCoeff();
    
    V1.rowwise() -= C;
    Vq.rowwise() -= C;

    const double max_span = BBOX.maxCoeff();
    V1 /= max_span;
    Vq /= max_span;
    
    BBOX = V1.colwise().maxCoeff()-V1.colwise().minCoeff();    
  }

  const size_t N = V1.rows();
  const std::vector<size_t> GROUP{0, N};
  
  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(V1);
  fps.compute('F');
  fps.reorder_geometry(V1, F1);
  fps.debug();
  {
    igl::writeOFF(string(OUTDIR+"/reorder_mesh.off"), V1, F1);
  }
  
  // GUI related
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu_config, menu_stat;  
  plugin.widgets.push_back(&menu_config);
  plugin.widgets.push_back(&menu_stat);  

  const Eigen::RowVector3d orange(1.0, 0.7, 0.2);  
  int SEL = 0;
  Eigen::RowVector3f last_mouse;

  // parameters
  double EPS = 0.0001;
  int   LAST_N = 20;
  float RHO = pt.get<double>("rho.value"), RHO_PREV = 0;
  double KL_ERROR = 0, PCG_ERROR = 0, PCG_ERROR_NO_PREC = 0;
  float MAX_NNZ_J = 0,   MEAN_NNZ_J = 0;
  float MAX_SUPER_J = 0, MEAN_SUPER_J = 0;
  int   VERT_NUM = V1.rows();
  PLOT_OPTION plot = LHSxX;
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_3D, KTYPE_PREV = static_cast<klchol::KERNEL_TYPE>(-1);
  SOLVER_TYPE SOL_TYPE = SOLVER_TYPE::KLCHOL;
  float EDGE_LENGTH = 0;
  float PAT_TIME = 0, CPY_TIME = 0, FAC_TIME = 0, SLV_TIME = 0, EVL_TIME = 0,
      PCG_TIME = 0, PCG_TIME_NO_PREC = 0, PRED_TIME = 0;
  float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
  int   NUM_SUPERNODES = 0;
  int   MAX_SUPERNODE_SIZE = 32;
  float WAVE_NUM = pt.get<double>("wave_num.value");  
  int   PCG_MAXITS = 7;
  int   NUM_SEC = 16;
  float HELM_K = 0;
  bool  NEAREST_K = false;

  // edge lengths
  VectorXd Le;
  igl::edge_lengths(V1, F1, Le);  
  EDGE_LENGTH = Le.sum()/Le.size();
  
  menu_config.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_config.menu_scaling(), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Config", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::InputDouble("EPS", &EPS, 0.001, 0.01);
    ImGui::SliderFloat("RHO", &RHO, 1.0f, 100.0f);
    ImGui::InputInt("supernode ID", &LAST_N, 1, 10000);
    ImGui::Combo("kernal", reinterpret_cast<int*>(&KTYPE), KERNEL_LIST);
    ImGui::Combo("solver", reinterpret_cast<int*>(&SOL_TYPE), "klchol\0ldlt\0\0");
    ImGui::SliderFloat("freq(rhs)", &WAVE_NUM, 0.0f, 100.0f);
    ImGui::SliderFloat("helmholtz k", &HELM_K, 0.0f, 100.0f);
    ImGui::Combo("plot", reinterpret_cast<int*>(&plot), "GreenFunc\0Sparsity\0lhs*x\0rhs\0length_scale\0\0");
    ImGui::InputInt("max supernode size", &MAX_SUPERNODE_SIZE, 1, 10000);
    ImGui::SliderInt("PCG maxits", &PCG_MAXITS, 1.0, 100);
    ImGui::Checkbox("Use nearest neighbors", &NEAREST_K);
    ImGui::End();
  };

  menu_stat.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_stat.menu_scaling(), 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::BulletText("vert num=%d", VERT_NUM);
    ImGui::BulletText("selected=%d", SEL);
    ImGui::BulletText("edge length=%f", EDGE_LENGTH);    
    ImGui::BulletText("bbox size=(%f %f %f)", BBOX.x(), BBOX.y(), BBOX.z());
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
    //    ImGui::BulletText("non-pcg error=%lf", PCG_ERROR_NO_PREC);
    ImGui::Text("---------------------------------------------");    
    ImGui::BulletText("time(calc pattern)   %.2f ms", PAT_TIME);
    ImGui::BulletText("time(data copy)      %.2f ms", CPY_TIME);
    ImGui::BulletText("time(factorize)      %.2f ms", FAC_TIME);
    ImGui::BulletText("time(solve)          %.2f ms", SLV_TIME);
    ImGui::BulletText("time(evalKx)         %.2f ms", EVL_TIME);
    ImGui::BulletText("time(pcg)            %.2f ms", PCG_TIME);
    //    ImGui::BulletText("time(non-pcg)        %.2f ms", PCG_TIME_NO_PREC);
    ImGui::BulletText("time(predict)        %.2f ms", PRED_TIME);
    ImGui::End();
  };

  // init the gpu cholesky solver
  const size_t DIM = 1;    
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON>(N, DIM, NUM_SEC));
  super_solver->set_source_points(N, V1.data());
  super_solver->set_target_points(Vq.rows(), Vq.data());

  mschol::high_resolution_timer timer;

  VectorXd PARAM = VectorXd::Zero(8), PARAM_PREV = PARAM;
  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  VectorXd Ky(N), rhs(N), y = VectorXd::Zero(N), plot_data(N);
  std::pair<int, double> pcg_ret;
  VectorXd cond_mean(Vq.rows());

  // // init kernel assembler
  // std::unique_ptr<klchol::cov_assembler<PDE_TYPE::POISSON>> ker_asm;
  // ker_asm.reset(new klchol::cov_assembler<PDE_TYPE::POISSON>(N));
  // ker_asm->set_source_pts(N, V1.data());
  // PARAM[0] = EPS;
  // PARAM[1] = HELM_K;
  // PARAM[2] = RHO;
  // PARAM[7] = KTYPE;  
  // ker_asm->assemble_kernel(KTYPE, PARAM.data(), PARAM.size());
  // ker_asm->debug(KTYPE);
  // exit(0);
 
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

  VectorXi sup_ptr, sup_ind, sup_parent;  
  const auto &update
      = [&]()
        {
          PARAM[0] = EPS;
          PARAM[1] = HELM_K;
          PARAM[2] = RHO;
          PARAM[7] = KTYPE;
          bool kernel_changed = ((PARAM-PARAM_PREV).norm() > 1e-6 ) || (KTYPE != KTYPE_PREV);
          if ( kernel_changed ) {
            super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
            PARAM_PREV = PARAM;
            KTYPE_PREV = KTYPE;
          }

          bool pattern_changed = (RHO != RHO_PREV);
          if ( pattern_changed ) {
            timer.start();
            {
              if ( NEAREST_K ) {
                fps.nearest_sparsity(RHO, DIM, PATT);
              } else {
                fps.simpl_sparsity(RHO, DIM, PATT); 
              }
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
            timer.stop();
            PAT_TIME = timer.duration();

            NUM_SUPERNODES = sup_ptr.size()-1;
            SIMPL_NNZ = PATT.nonZeros();
            SUPER_NNZ = SUP_PATT.nonZeros();
            PARAM[6] = 1.0*SUPER_NNZ/SUP_PATT.size();

#if 1
            if ( SUP_PATT.rows() < 20000 ) {
              PermutationMatrix<-1, -1> Q(N);
              for (size_t j = 0; j < N; ++j) {
                Q.indices()[sup_ind[j]] = j;
              }
              const SparseMatrix<double> tmpP = SUP_PATT*Q.transpose();

              char outf[256];              
              sprintf(outf, "%s/super-re-patt-%lf.mat", OUTDIR.c_str(), RHO);
              mschol::write_sparse_matrix(outf, tmpP);
              sprintf(outf, "%s/simpl-patt-%lf.mat", OUTDIR.c_str(), RHO);
              mschol::write_sparse_matrix(outf, PATT);
              sprintf(outf, "%s/super-patt-%lf.mat", OUTDIR.c_str(), RHO);
              mschol::write_sparse_matrix(outf, SUP_PATT);              
            }
#endif

            timer.start();
            {
              super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
              super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
            }
            timer.stop();
            CPY_TIME = timer.duration();

            THETA_NNZ = super_solver->theta_nnz();

            RHO_PREV = RHO;
            analyse_nnz(SUP_PATT.cols(), SUP_PATT.outerIndexPtr(), MAX_NNZ_J, MEAN_NNZ_J);
            analyse_nnz(sup_ptr.size()-1, &sup_ptr[0], MAX_SUPER_J, MEAN_SUPER_J);
          }

          if ( kernel_changed || pattern_changed ) {
            timer.start();
            {              
              super_solver->compute();
            }
            timer.stop();
            FAC_TIME = timer.duration();
          }

          // rhs
          rhs.setZero();
          {
            Eigen::RowVector3d Omega;
            Omega.x() = BBOX.x() < 1e-8 ? 0 : 1/BBOX.x()*2*M_PI*WAVE_NUM;
            Omega.y() = BBOX.y() < 1e-8 ? 0 : 1/BBOX.y()*2*M_PI*WAVE_NUM;
            Omega.z() = BBOX.z() < 1e-8 ? 0 : 1/BBOX.z()*2*M_PI*WAVE_NUM;
            for (size_t p = 0; p < V1.rows(); ++p) {
              rhs[p] = 
                  cos(Omega.x()*V1(p, 0))*
                  cos(Omega.y()*V1(p, 1))*
                  cos(Omega.z()*V1(p, 2));
            }
          }

          // y.setZero();
          // timer.start();
          // {                        
          //   super_solver->solve(rhs.data(), y.data());
          // }
          // timer.stop();
          // SLV_TIME = timer.duration();
          // spdlog::info("target norm={0:.8f}", rhs.norm());
          // spdlog::info("source norm={0:.8f}", y.norm());

          // Ky.setZero();
          // timer.start();
          // {
          //   super_solver->evalKx(Ky.data());
          // }
          // timer.stop();
          // EVL_TIME = timer.duration();
          // KL_ERROR = (Ky-rhs).norm()/rhs.norm();

          // spdlog::info("sparsity={0:d}, {1:.06f}, {2:.06f}", NEAREST_K, 1.0*SUP_PATT.nonZeros()/SUP_PATT.size(), KL_ERROR);

          // try PCG with preconditioner
          y.setZero();
          timer.start();
          {
            pcg_ret = super_solver->pcg(rhs.data(), y.data(), true, 1000, pt.get<double>("tol.value"));
          }
          timer.stop();
          PCG_TIME = timer.duration();
          PCG_ERROR = pcg_ret.second;

          // // try PCG without preconditioner
          // y.setZero();
          // timer.start();
          // {
          //   pcg_ret = super_solver->pcg(rhs.data(), y.data(), false, 10000, PCG_ERROR);
          // }
          // timer.stop();
          // PCG_TIME_NO_PREC = timer.duration();
          // PCG_ERROR_NO_PREC = pcg_ret.second;
          
          // visualize the data

          // cond_mean.setZero();
          // timer.start();
          // {
          //   super_solver->predict(1, y.data(), y.size(), cond_mean.data(), cond_mean.size());
          // }
          // timer.stop();
          // PRED_TIME = timer.duration();
          
          // if ( plot == PLOT_OPTION::LHSxX ) {
          //   plot_data = Ky;
          // } else if ( plot == PLOT_OPTION::RHS ) {
          //   plot_data = rhs;
          // } else if ( plot == PLOT_OPTION::SPARSITY ) {
          //   plot_data = SparseMatrix<double>(PATT.selfadjointView<Eigen::Lower>()).col(SEL);
          // } else if ( plot == PLOT_OPTION::GREEN_FUNC ) {
          //   // for (size_t pid = 0; pid < V1.rows(); ++pid) {
          //   //   super_solver->kernel<Eigen::RowVector3d>(V1.row(pid), V1.row(SEL), PARAM.data(), &plot_data[pid]);         
          //   // }
          // } else if ( plot == PLOT_OPTION::SCALE ) {
          //   const auto L_sel = fps.get_len_scale(SEL);
          //   for (size_t pid = 0; pid < V1.rows(); ++pid) {
          //     if ( (V1.row(pid)-V1.row(SEL)).squaredNorm() <= L_sel*L_sel ) {
          //       plot_data[pid] = 1;
          //     }
          //   }
          // }
          viewer.data().set_data(rhs, igl::COLOR_MAP_TYPE_PARULA);
          spdlog::info("pcg error={0:d} {1:.6f}", pcg_ret.first, PCG_ERROR);
          
          // // write out supernode infos
          // {
          //   char outfile[256];
          //   sprintf(outfile, "%s/super_parent_rho_%lf.dat", OUTDIR.c_str(), RHO);
          //   ofstream ofs(outfile);
          //   ofs.write(reinterpret_cast<char *>(sup_parent.data()), sup_parent.size()*sizeof(sup_parent[0]));
          //   ofs.close();
          // }

          // // visualize last N points
          // //          viewer.data().set_points(V1.bottomRows(LAST_N), orange);
          // // visualize supernodes
          // // LAST_N = std::min(LAST_N, static_cast<int>(sup_ptr.size()-2));
          // // const VectorXi &sup_idx = sup_ind.segment(sup_ptr[LAST_N], sup_ptr[LAST_N+1]-sup_ptr[LAST_N]);
          // // viewer.data().set_points(V1(sup_idx.array(), Eigen::all), orange);
          // if ( LAST_N >= 0 ) {
          //   viewer.data().set_points(V1.topRows(LAST_N), Eigen::RowVector3d(1.0, 0.0, 0.0));
          // } else {
          //   viewer.data().set_points(V1.bottomRows(-LAST_N), Eigen::RowVector3d(1.0, 0.0, 0.0));
          // }
        };
      
  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int, int) -> bool {
      last_mouse = Eigen::RowVector3f(
      viewer.current_mouse_x,viewer.core().viewport(3)-viewer.current_mouse_y,0);

      // Find closest point on mesh to mouse position
      int fid;
      Eigen::Vector3f bary;
      if(igl::unproject_onto_mesh(
             last_mouse.head(2),
             viewer.core().view,
             viewer.core().proj, 
             viewer.core().viewport, 
             V1, F1, 
             fid, bary))
      {
        long c;
        bary.maxCoeff(&c);
        SEL = F1(fid, c);
        update();
        return true;
      }
      return false;
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
            break;
          case 's':
          case 'S':
            spdlog::info("save mesh and data {} times", SAVE_COUNT);
            save_mesh_and_data(OUTDIR, "mesh", V1, F1, "solution", y, "param", PARAM);
            save_mesh_and_data(OUTDIR, "query-mesh", Vq, Fq, "query-solution", cond_mean, "query-param", PARAM);
            ++SAVE_COUNT;
            break;
          default:
            return false;
        }
        return true;
      };  

  viewer.data().set_mesh(V1, F1);
  viewer.data().show_lines = false;
  viewer.core().align_camera_center(V1, F1);
  viewer.core().background_color.setOnes();
  viewer.launch();
  
  return 0;
}
