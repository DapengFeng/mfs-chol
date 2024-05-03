#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <imgui.h>
#include <iostream>
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#include <spdlog/spdlog.h>

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/timer.h"
#include "src/io.h"
#include "src/ptree.h"
#include "src/gputimer.h"

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

#if 0
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
#endif

template <class Vec>
static int save_mesh_and_data(const string &mesh_file,
                              const RmMatF_t &V,
                              const RmMatI_t &F,
                              const string &data_file,
                              const Vec &data)
{
  typedef typename Vec::Scalar Scalar;

  // write mesh
  igl::writeOBJ(mesh_file, V, F);

  // write solutions
  ofstream ofs(data_file, ios::binary);
  ofs.write((char*)data.data(), data.size()*sizeof(Scalar));
  ofs.close();
  
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

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  RmMatF_t Vs, Vq, VV;    // source, targets and merged mesh
  RmMatI_t Fs, Fq, FF;    // faces

  const string MESH       = pt.get<string>("source_mesh.value");
  const string QUERY_MESH = pt.get<string>("target_mesh.value");
  const string OUTDIR     = pt.get<string>("outdir.value");  

  // read meshes
  igl::readOFF(MESH, Vs, Fs);
  igl::readOFF(QUERY_MESH, Vq, Fq);
  spdlog::info("source mesh={}", Vs.rows());
  spdlog::info("target mesh={}", Vq.rows());
  
  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(Vs);
  fps.compute('F');
  fps.reorder_geometry(Vs, Fs);
  fps.debug();
  {
    igl::writeOFF(string(OUTDIR+"/reorder_mesh.off"), Vs, Fs);
  }

  // merge mesh
  RowVector3d BBOX, MIN_COORD, MAX_COORD;
  merge_meshes(Vq, Fq, Vs, Fs, VV, FF);
  const Eigen::RowVector3d C_s = Vs.colwise().sum()/Vs.rows();
  Eigen::RowVector3d MIN_COORD_s, MAX_COORD_s, BBOX_s;
  double MAX_SPAN;
  {
    // put source to the center
    Vs.rowwise() -= C_s;
    Vq.rowwise() -= C_s;   
    VV.rowwise() -= C_s;

    // get bounding box of the target mesh
    Eigen::RowVector3d BBOX_q = Vq.colwise().maxCoeff()-Vq.colwise().minCoeff();    
    MAX_SPAN = BBOX_q.maxCoeff();
    Vs /= MAX_SPAN;
    Vq /= MAX_SPAN;
    VV /= MAX_SPAN;

    // min max coord of the merged mesh
    MIN_COORD = VV.colwise().minCoeff();
    MAX_COORD = VV.colwise().maxCoeff();
    BBOX = MAX_COORD-MIN_COORD;

    // min max coord of the source mesh
    MIN_COORD_s = Vs.colwise().minCoeff();
    MAX_COORD_s = Vs.colwise().maxCoeff();
    BBOX_s = MAX_COORD_s-MIN_COORD_s;
  }
  std::cout << "\tMIN_COORD=" << MIN_COORD << std::endl;
  std::cout << "\tMAX_COORD=" << MAX_COORD << std::endl;
  std::cout << "\tBBOX=" << BBOX << std::endl;

  // PBBFMM data storage
  std::vector<vector3> all_points(VV.rows());
  for (size_t i = 0; i < all_points.size(); ++i) {
    all_points[i].x = VV(i, 0);
    all_points[i].y = VV(i, 1);
    all_points[i].z = VV(i, 2);      
  }  
  
  // GUI related
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu_config, menu_stat;  
  plugin.widgets.push_back(&menu_config);
  plugin.widgets.push_back(&menu_stat);  

  // parameters
  double EPS = pt.get<double>("eps.value");
  float  RHO = pt.get<float>("rho.value");
  double KL_ERROR = 0, PCG_ERROR = 0, PCG_ERROR_NO_PREC = 0;
  float MAX_NNZ_J = 0,   MEAN_NNZ_J = 0;
  float MAX_SUPER_J = 0, MEAN_SUPER_J = 0;
  int   VERT_NUM = Vs.rows();
  PLOT_OPTION plot = LHSxX;
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_3D;
  float EDGE_LENGTH = 0;
  float PAT_TIME = 0, CPY_TIME = 0, FAC_TIME = 0, SLV_TIME = 0, EVL_TIME = 0,
      PCG_TIME = 0, PCG_TIME_NO_PREC = 0, PRED_TIME = 0, FMM_TIME = 0;
  float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
  int   NUM_SUPERNODES = 0;
  int   MAX_SUPERNODE_SIZE = pt.get<int>("max_supernode_size.value");
  float WAVE_NUM = pt.get<float>("wave_num.value", 0);
  int   PCG_MAXITS = pt.get<int>("pcg_iters.value");
  int   NUM_SEC = pt.get<int>("num_sec.value");
  bool  CUDA_EXTRP = false;
  bool  JACOBI_PRECOND = false;
  bool  RESET_CHARGE = pt.get<bool>("reset_charge.value", false);
  bool  USE_PCG = pt.get<bool>("use_pcg.value", true);
  spdlog::info("wave number={0:.3f}", WAVE_NUM);
  bool  USE_GUI = pt.get<bool>("use_gui.value", false);
  bool  REVERT_TRANSFORM = pt.get<bool>("revert_transform.value", false);
  
  // edge lengths
  VectorXd Le;
  igl::edge_lengths(Vs, Fs, Le);  
  EDGE_LENGTH = Le.sum()/Le.size();
  spdlog::info("mean edge length={0:.6f}", EDGE_LENGTH);
  
  menu_config.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_config.menu_scaling(), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Config", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::InputDouble("EPS", &EPS, 0.001, 0.01);
    ImGui::SliderFloat("RHO", &RHO, 1.0f, 100.0f);
    ImGui::Combo("kernal", reinterpret_cast<int*>(&KTYPE), KERNEL_LIST);
    ImGui::SliderFloat("freq(rhs)", &WAVE_NUM, 0.0f, 100.0f);
    ImGui::Combo("plot", reinterpret_cast<int*>(&plot), "GreenFunc\0Sparsity\0lhs*x\0rhs\0length_scale\0\0");
    ImGui::InputInt("max supernode size", &MAX_SUPERNODE_SIZE, 1, 10000);
    ImGui::SliderInt("PCG maxits", &PCG_MAXITS, 1.0, 100);
    ImGui::Checkbox("cuda extrap", &CUDA_EXTRP);
    ImGui::Checkbox("Jacobi precond", &JACOBI_PRECOND);
    ImGui::Checkbox("use_pcg", &USE_PCG);
    ImGui::Checkbox("reset_charge", &RESET_CHARGE);
    ImGui::End();
  };

  menu_stat.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_stat.menu_scaling(), 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::BulletText("vert num=%d", VERT_NUM);
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
    ImGui::BulletText("time(FMM)            %.2f ms", FMM_TIME);
    ImGui::End();
  };

  // read charges
  const string CHARGE_FILE = pt.get<string>("source_charge.value");
  VectorXd rhs(Vs.rows());
  {
    std::ifstream ifs(CHARGE_FILE, std::ios_base::binary);
    if ( ifs.fail() ) {
      cerr << "# no charge file!" << endl;
      rhs.setOnes(); // for now
    } else {
      ifs.seekg(0, std::ios_base::end);
      size_t length = ifs.tellg()/sizeof(double);
      ifs.seekg(0, std::ios_base::beg);
      ASSERT(length == Vs.rows());
    
      ifs.read(reinterpret_cast<char*>(rhs.data()), length*sizeof(double));
      ifs.close();
    }
    rhs = fps.P()*rhs;    
  }
  
  // init the gpu cholesky solver
  const size_t DIM = 1;
  const size_t N = Vs.rows();
  const std::vector<size_t> GROUP{0, N};
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON>(N, DIM, NUM_SEC));
  super_solver->set_source_points(Vs.rows(), Vs.data());
  super_solver->set_target_points(VV.rows(), VV.data());

  GpuTimer gpu_timer;

  VectorXd PARAM = VectorXd::Zero(8);  
  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  VectorXd y = VectorXd::Zero(N);
  VectorXd fmm_potent = VectorXd::Zero(VV.rows());  

  FMM_parameters fmm_config;
  {
    // fmm_config.L                   = 2*max(MIN_COORD.cwiseAbs().maxCoeff(), MAX_COORD.cwiseAbs().maxCoeff());
    fmm_config.L                   = 2*max(MIN_COORD_s.cwiseAbs().maxCoeff(), MAX_COORD_s.cwiseAbs().maxCoeff());    
    fmm_config.tree_level          = Vs.rows() > 180000 ? 6 : 5; // emperical
    fmm_config.interpolation_order = 4;
    fmm_config.eps                 = 5e-5;
    fmm_config.use_chebyshev       = 1;
    fmm_config.reg_eps             = EPS;
  }
  cout << "\tFMM.L=" << fmm_config.L << endl;
 
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
          PARAM[2] = RHO;
          PARAM[7] = KTYPE;
          super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
          spdlog::info("rho={0:.3f}, eps={1:.6f}", RHO, EPS);
            
          gpu_timer.start();
          {
            fps.simpl_sparsity(RHO, DIM, PATT);
            fps.aggregate(DIM, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
            fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);
          }
          gpu_timer.stop();
          PAT_TIME = gpu_timer.elapsed();
          spdlog::info("TIME(pattern)={0:.3f}", gpu_timer.elapsed());

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

          THETA_NNZ = super_solver->theta_nnz();
          spdlog::info("momery cost={0:.6f}", super_solver->memory());

          analyse_nnz(SUP_PATT.cols(), SUP_PATT.outerIndexPtr(), MAX_NNZ_J, MEAN_NNZ_J);
          analyse_nnz(sup_ptr.size()-1, &sup_ptr[0], MAX_SUPER_J, MEAN_SUPER_J);

          gpu_timer.start();
          {              
            super_solver->compute();
          }
          gpu_timer.stop();
          FAC_TIME = gpu_timer.elapsed();
          spdlog::info("TIME(factorize)={0:.3f}", gpu_timer.elapsed());

          if ( RESET_CHARGE ) {
            for (size_t i = 0; i < rhs.size(); ++i) {
              rhs[i] =
                  cos(WAVE_NUM*Vs(i, 0))*
                  cos(WAVE_NUM*Vs(i, 1))*
                  cos(WAVE_NUM*Vs(i, 2));
              rhs[i] = rhs[i] > 0 ? 1 : -1;
            }
          }

          if ( USE_PCG ) {
            // pcg with precond.
            gpu_timer.start();
            {
              y.setZero();            
              auto pcg_ret = super_solver->pcg(rhs.data(), y.data(), true, PCG_MAXITS, 1e-6, &fmm_config);
              PCG_ERROR = pcg_ret.second;          
            }
            gpu_timer.stop();
            PCG_TIME = gpu_timer.elapsed();
            spdlog::info("TIME(pcg)={0:.3f}", PCG_TIME);
            spdlog::info("pcg error={0:.6f}", PCG_ERROR);
          } else {
            // direct solver
            gpu_timer.start();
            {
              y.setZero();
              super_solver->solve(rhs.data(), y.data());
            }
            gpu_timer.stop();
            const double DIRECT_TIME = gpu_timer.elapsed();
            spdlog::info("TIME(direct)={0:.3f}", DIRECT_TIME);

            // evaluate for computing error
            VectorXd Ky = VectorXd::Zero(rhs.size());
            {
              super_solver->evalKx(Ky.data());
            }
            const double DIRECT_ERROR = (Ky-rhs).norm()/rhs.norm();
            spdlog::info("direct error={0:.6f}", DIRECT_ERROR);
          }

          // fmm extrapolation
          VectorXd fmm_res = VectorXd::Zero(VV.rows());
          fmm_res.tail(Vs.rows()) = y;
          fmm_potent.setZero();
          gpu_timer.start();
          {
            fmm_config.L = 2*max(MIN_COORD.cwiseAbs().maxCoeff(), MAX_COORD.cwiseAbs().maxCoeff());            
            std::vector<std::shared_ptr<H2_3D_Tree>> trees;    
            klchol::create_fmm_tree(trees, KTYPE, fmm_config);
            H2_3D_Compute<H2_3D_Tree> compute
                (*trees[0], all_points, all_points, fmm_res.data(), 1, fmm_potent.data());
          }
          gpu_timer.stop();
          FMM_TIME = gpu_timer.elapsed();
          spdlog::info("FMM time={0:.3f}", FMM_TIME);

          if ( JACOBI_PRECOND ) {
            y.setZero();
            auto pcg_ret = super_solver->pcg(rhs.data(), y.data(), false, 100, PCG_ERROR, &fmm_config);
            spdlog::info("non-pcg iter={}", pcg_ret.first);
            spdlog::info("non-pcg error={0:.6f}", pcg_ret.second); 
          }
          
          if ( CUDA_EXTRP ) {
            VectorXd cuda_potent = VectorXd::Zero(fmm_potent.size());
            gpu_timer.start();
            {
              super_solver->predict(
                  1, y.data(), y.size(),
                  cuda_potent.data(), cuda_potent.size());
            }
            gpu_timer.stop();
            PRED_TIME = gpu_timer.elapsed();
            spdlog::info("cuda prediction time={0:.3f}", PRED_TIME);
            spdlog::info("FMM error={0:.6f}", (fmm_potent-cuda_potent).norm()/cuda_potent.norm());            
          }

          if ( USE_GUI ) {
            // visualize solved potential
            viewer.data().set_data(fmm_potent, igl::COLOR_MAP_TYPE_PARULA);
          } else {
            // save mesh and data
            cout << "\tsave mesh and data..." << endl;
            if ( REVERT_TRANSFORM ) {
              cout << "# revert transformation.." << endl;
              VV *= MAX_SPAN;
              VV.rowwise() += C_s;
            }            
            save_mesh_and_data(
                pt.get<string>("out_mesh.value"), VV, FF,
                pt.get<string>("out_data.value"), fmm_potent);            
          }
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
            {
              if ( REVERT_TRANSFORM ) {
                cout << "# revert transformation.." << endl;
                VV *= MAX_SPAN;
                VV.rowwise() += C_s;
              }
              save_mesh_and_data(
                  pt.get<string>("out_mesh.value"), VV, FF,
                  pt.get<string>("out_data.value"), fmm_potent);
            }
            break;
          default:
            return false;
        }
        return true;
      };  

  if ( USE_GUI ) {
    viewer.data().set_mesh(VV, FF);
    viewer.data().show_lines = false;
    viewer.core().align_camera_center(VV, FF);
    viewer.launch();
  } else {
    update();
  }
  
  return 0;
}
