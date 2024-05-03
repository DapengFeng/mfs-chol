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

#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/timer.h"
#include "src/io.h"
#include "src/gputimer.h"
#include "src/ptree.h"

using namespace std;
using namespace Eigen;
using namespace klchol;

enum PLOT_OPTION {
  GREEN_FUNC,
  SPARSITY,
  LHSxX,
  RHS,
  SCALE
};

enum SOLVER_TYPE {
  KLCHOL,
  LDLT
};

static int SAVE_COUNT = 1;

template <class Vec1, class Vec2>
static int save_mesh_and_data(const std::string &dir,
                              const string &mesh_name,
                              const RmMatF_t &V,
                              const RmMatI_t &F,
                              const string &data_name,
                              const Vec1 &data,
                              const string &param_name,
                              const Vec2 &eps) 
{
  char outf[256];

  // write mesh
  sprintf(outf, "%s/%s-%03d.obj", dir.c_str(), mesh_name.c_str(), SAVE_COUNT);
  igl::writeOBJ(outf, V, F);

  // write solutions
  sprintf(outf, "%s/%s-%03d.dat", dir.c_str(), data_name.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)data.data(), data.size()*sizeof(typename Vec1::Scalar));
    ofs.close();
  }

  // write parameters
  sprintf(outf, "%s/%s-%03d.dat", dir.c_str(), param_name.c_str(), SAVE_COUNT);
  {
    ofstream ofs(outf, ios::binary);
    ofs.write((char*)eps.data(), eps.size()*sizeof(typename Vec2::Scalar));
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

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  RmMatF_t Vs, Vd, Vq, VV;  // vertices and normals
  RmMatI_t Fs, Fd, Fq, FF;  // faces

  const string SOURCE_MESH = pt.get<string>("source_mesh.value");
  const string QUERY_MESH  = pt.get<string>("target_mesh.value");
  const string DEFORM_MESH = pt.get<string>("deformed_source_mesh.value");
  const string OUTDIR      = pt.get<string>("outdir.value");

  // read the mesh
  igl::readOFF(SOURCE_MESH, Vs, Fs);
  igl::readOFF(QUERY_MESH, Vq, Fq);
  const bool has_deform_mesh = igl::readOFF(DEFORM_MESH, Vd, Fd);
  if ( has_deform_mesh ) {
    cout << "\tthere is deformed source" << endl;
    cout << Vd.rows() << " " << Vs.rows() << endl;
    cout << Fd.rows() << " " << Fs.rows() << endl;
    ASSERT(Vd.rows() == Vs.rows() && Fd.rows() == Fs.rows());
  }

  GpuTimer gpu_timer;

  // compute FPS for maxmin-ordering, a.k.a. coarse to fine
  fps_sampler fps(Vs);
  gpu_timer.start();
  {
    fps.compute('F');    
  }
  gpu_timer.stop();
  const double FPS_TIME = gpu_timer.elapsed()/1000;
  fps.reorder_geometry(Vs, Fs);
  fps.debug();

  Eigen::RowVector3d BBOX, MIN_COORD, MAX_COORD;  
  merge_meshes(Vq, Fq, Vs, Fs, VV, FF);
  {
    // centralize and normalize meshes with target mesh
    const Eigen::RowVector3d C = VV.colwise().sum()/VV.rows();
    BBOX = VV.colwise().maxCoeff()-VV.colwise().minCoeff();

    Vs.rowwise() -= C;
    Vd.rowwise() -= C;
    Vq.rowwise() -= C;   
    VV.rowwise() -= C;

    const double max_span = BBOX.maxCoeff();
    Vs /= max_span;
    Vd /= max_span;
    Vq /= max_span;
    VV /= max_span;
    
    MIN_COORD = VV.colwise().minCoeff();
    MAX_COORD = VV.colwise().maxCoeff();
    BBOX = MAX_COORD-MIN_COORD;
    cout << "\tmax coord=" << MIN_COORD << endl;
    cout << "\tmin coord=" << MAX_COORD << endl;
    cout << "\tbbox=" << BBOX << endl;
    spdlog::info("Vq rows={}", Vq.rows());
  }
  spdlog::info("source pts={}, target pts={}", Vs.rows(), Vq.rows());

  igl::writeOFF(string(OUTDIR+"/target_rest.off"), Vq, Fq);
  igl::writeOFF(string(OUTDIR+"/source_rest.off"), Vs, Fs);
  igl::writeOFF(string(OUTDIR+"/source_deform.off"), Vd, Fd);

  // PBBFMM data
  std::vector<vector3> all_points(VV.rows());
  for (size_t i = 0; i < all_points.size(); ++i) {
    all_points[i].x = VV(i, 0);
    all_points[i].y = VV(i, 1);
    all_points[i].z = VV(i, 2);    
  }

  const size_t N = Vs.rows();
  const std::vector<size_t> GROUP{0, N};
    
  // GUI related
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu_config, menu_stat;  
  plugin.widgets.push_back(&menu_config);
  plugin.widgets.push_back(&menu_stat);  

  const Eigen::RowVector3d orange(1.0, 0.7, 0.2);  
  //  int SEL = 0;
  Eigen::RowVector3f last_mouse;

  // parameters
  double EPS = pt.get<double>("eps.value");
  int   LAST_N = 20;
  float RHO = pt.get<double>("rho.value"), RHO_PREV = 0;
  float KL_ERROR = 0, PCG_ERROR = 0;
  float MAX_NNZ_J = 0,   MEAN_NNZ_J = 0;
  float MAX_SUPER_J = 0, MEAN_SUPER_J = 0;
  int   VERT_NUM = Vs.rows();
  PLOT_OPTION plot = LHSxX;
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::KELVIN_3D, KTYPE_PREV = static_cast<klchol::KERNEL_TYPE>(-1);
  SOLVER_TYPE SOL_TYPE = SOLVER_TYPE::KLCHOL;
  float EDGE_LENGTH = 0;
  float PAT_TIME = 0, CPY_TIME = 0, FAC_TIME = 0, SLV_TIME = 0, EVL_TIME = 0, PCG_TIME = 0, TIL_TIME = 0, PRED_TIME = 0;
  float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
  int   NUM_SUPERNODES = 0;
  int   MAX_SUPERNODE_SIZE = 100000000;
  float WAVE_NUM = 0;
  int   PCG_MAXITS = pt.get<int>("pcg_iters.value");
  float MU = 1, NU = pt.get<double>("nu.value", 0.3);
  VectorXd PARAM = VectorXd::Zero(8), PARAM_PREV = PARAM;
  int NUM_SEC = pt.get<int>("sec_num.value");
  int THREADS = 32;
  bool USE_GUI = pt.get<bool>("use_gui.value", true);

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
    ImGui::InputInt("supernode ID", &LAST_N, 1, 10000);
    ImGui::Combo("kernal", reinterpret_cast<int*>(&KTYPE), KERNEL_LIST);
    ImGui::Combo("solver", reinterpret_cast<int*>(&SOL_TYPE), "klchol\0ldlt\0\0");
    ImGui::SliderFloat("wave number", &WAVE_NUM, 0.0f, 100.0f);
    ImGui::SliderFloat("mu", &MU, 1.0f, 1000.0f);
    ImGui::SliderFloat("nu", &NU, 0.0f, 0.49f);
    ImGui::Combo("plot", reinterpret_cast<int*>(&plot), "GreenFunc\0Sparsity\0lhs*x\0rhs\0length_scale\0\0");
    ImGui::InputInt("max supernode size", &MAX_SUPERNODE_SIZE, 1, 10000);
    ImGui::SliderInt("PCG maxits", &PCG_MAXITS, 1.0, 100);
    ImGui::SliderInt("threads", &THREADS, 16, 256);
    ImGui::End();
  };

  menu_stat.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowPos(ImVec2(180.f*menu_stat.menu_scaling(), 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::BulletText("vert num=%d", VERT_NUM);
    //    ImGui::BulletText("selected=%d", SEL);
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
    ImGui::Text("---------------------------------------------");    
    ImGui::BulletText("time(calc pattern)   %.2f ms", PAT_TIME);
    ImGui::BulletText("time(data copy)      %.2f ms", CPY_TIME);
    ImGui::BulletText("time(factorize)      %.2f ms", FAC_TIME);
    ImGui::BulletText("time(solve)          %.2f ms", SLV_TIME);
    ImGui::BulletText("time(evalKx)         %.2f ms", EVL_TIME);
    ImGui::BulletText("time(evalKx_tiled)   %.2f ms", TIL_TIME);    
    ImGui::BulletText("time(pcg)            %.2f ms", PCG_TIME);
    ImGui::BulletText("time(predict)        %.2f ms", PRED_TIME);
    ImGui::End();
  };
    
  // init the gpu cholesky solver
  const size_t DIM = 3;
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::KELVIN>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::KELVIN>(N, DIM, NUM_SEC));
  super_solver->set_source_points(N, Vs.data());
  super_solver->set_target_points(Vq.rows(), Vq.data());

  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  RmMatF_t Ky(N, 3), rhs(N, 3);
  VectorXd y = VectorXd::Zero(3*N);
  VectorXd plot_data(N);
  std::pair<int, double> pcg_ret;
  RmMatF_t cond_mean(Vq.rows(), 3);

  if ( has_deform_mesh ) {
    rhs = fps.P()*Vd-Vs;
  }
  
  FMM_parameters fmm_config;
  {
    fmm_config.L                   = 2*max(MIN_COORD.cwiseAbs().maxCoeff(), MAX_COORD.cwiseAbs().maxCoeff());
    fmm_config.tree_level          = Vs.rows() > 180000 ? 6 : 5; // emperical
    fmm_config.interpolation_order = 4;
    fmm_config.eps                 = 1e-6;
    fmm_config.use_chebyshev       = 1;
    fmm_config.reg_eps             = EPS;
    fmm_config.a                   = 1/(4*M_PI*MU);
    fmm_config.b                   = fmm_config.a/(4-4*NU);
  }
  spdlog::info("fmm L={0:.6f}, eps={1:.6f}, a={2:.6f}, b={3:.6f}", fmm_config.L, fmm_config.reg_eps, fmm_config.a, fmm_config.b);

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
  RmMatF_t fmm_res = RmMatF_t::Zero(VV.rows(), 3);  
  const auto &update
      = [&]()
        {
          PARAM[0] = EPS;
          PARAM[1] = 1/(4*M_PI*MU);
          PARAM[2] = PARAM[1]/(4-4*NU);
          PARAM[3] = RHO;
          PARAM[7] = KTYPE;
          bool kernel_changed = ((PARAM-PARAM_PREV).norm() > 1e-6) || (KTYPE != KTYPE_PREV);
          if ( kernel_changed ) {
            super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
            PARAM_PREV = PARAM;
            KTYPE_PREV = KTYPE;
          }

          bool pattern_changed = (RHO != RHO_PREV);
          if ( pattern_changed ) {
            gpu_timer.start();
            {
              fps.simpl_sparsity(RHO, DIM, PATT);
              fps.aggregate(DIM, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
              fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);
            }
            gpu_timer.stop();
            PAT_TIME = gpu_timer.elapsed()/1000;
            spdlog::info("TIME(patt)={0:.3f}", PAT_TIME);

            NUM_SUPERNODES = sup_ptr.size()-1;
            SIMPL_NNZ = PATT.nonZeros();
            SUPER_NNZ = SUP_PATT.nonZeros();
            PARAM[6] = 1.0*SUPER_NNZ/SUP_PATT.size();
            spdlog::info("patt size={}", PATT.rows());

            gpu_timer.start();
            {
              super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
              super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
            }
            gpu_timer.stop();
            CPY_TIME = gpu_timer.elapsed()/1000;
            spdlog::info("TIME(cpy)={0:.3f}", CPY_TIME);

            THETA_NNZ = super_solver->theta_nnz();

            RHO_PREV = RHO;
            analyse_nnz(SUP_PATT.cols(), SUP_PATT.outerIndexPtr(), MAX_NNZ_J, MEAN_NNZ_J);
            analyse_nnz(sup_ptr.size()-1, &sup_ptr[0], MAX_SUPER_J, MEAN_SUPER_J);
          }

          spdlog::info("TIME(precompute)={0:.3f}", FPS_TIME+PAT_TIME+CPY_TIME);

          if ( kernel_changed || pattern_changed ) {
            gpu_timer.start();
            {              
              super_solver->compute();
            }
            gpu_timer.stop();
            FAC_TIME = gpu_timer.elapsed()/1000;
            spdlog::info("TIME(compute)={0:.3f}", FAC_TIME);
          }

          // rhs
          if ( !has_deform_mesh ) {
            rhs.setZero();
            {
              Eigen::RowVector3d Omega;
              Omega.x() = BBOX.x() < 1e-8 ? 0 : 1/BBOX.x()*2*M_PI*WAVE_NUM;
              Omega.y() = BBOX.y() < 1e-8 ? 0 : 1/BBOX.y()*2*M_PI*WAVE_NUM;
              Omega.z() = BBOX.z() < 1e-8 ? 0 : 1/BBOX.z()*2*M_PI*WAVE_NUM;
              for (size_t p = 0; p < Vs.rows(); ++p) {
                rhs.row(p) = 0.05*
                    cos(Omega.x()*Vs(p, 0))*
                    cos(Omega.y()*Vs(p, 1))*
                    cos(Omega.z()*Vs(p, 2))*Vs.row(p).normalized();
              }
            }
          }

          // try PCG
          gpu_timer.start();
          {
            y.setZero();
            pcg_ret = super_solver->pcg(rhs.data(), y.data(), true, PCG_MAXITS, 1e-6, &fmm_config);
          }
          gpu_timer.stop();
          PCG_TIME = gpu_timer.elapsed()/1000;
          PCG_ERROR = pcg_ret.second;
          spdlog::info("pcg error={0:.6f}", PCG_ERROR);
          spdlog::info("TIME(pcg)={0:.3f}", PCG_TIME);

          // Ky.setZero();
          // gpu_timer.start();
          // {
          //   super_solver->evalKx(Ky.data());
          // }
          // gpu_timer.stop();
          // EVL_TIME = gpu_timer.elapsed();
          // KL_ERROR = (Ky-rhs).norm()/rhs.norm();
          // spdlog::info("cuda eval time={0:.3f}", EVL_TIME);

          RmMatF_t weights = RmMatF_t::Zero(VV.rows(), 3);
          fmm_res.setZero();
          weights.bottomRows(Vs.rows()) = Eigen::Map<RmMatF_t>(y.data(), y.size()/3, 3);

          gpu_timer.start();
          {
            std::vector<std::shared_ptr<H2_3D_Tree>> tree;    
            klchol::create_fmm_tree(tree, KTYPE, fmm_config);            

            const int nn = VV.rows();

            // reorder charges in consecutive memory [xxx yyy zzz] and [xxx zzz]
            std::vector<double> charge_xyz(3*nn), charge_xz(2*nn);
            for (size_t i = 0; i < nn; ++i) {
              charge_xyz[0*nn+i] = *(weights.data()+3*i+0);
              charge_xyz[1*nn+i] = *(weights.data()+3*i+1);
              charge_xyz[2*nn+i] = *(weights.data()+3*i+2);

              charge_xz[0*nn+i]  = *(weights.data()+3*i+0);
              charge_xz[1*nn+i]  = *(weights.data()+3*i+2);
            }

            std::vector<double> output0_xx(nn, 0), output3_yy(nn, 0), output5_zz(nn, 0);
            std::vector<double> output1_xy(2*nn, 0), output2_xz(2*nn, 0), output4_yz(2*nn, 0);

            // xx
            H2_3D_Compute<H2_3D_Tree> compute0
                (*tree[0], all_points, all_points, &charge_xyz[0*nn], 1, &output0_xx[0]);
            // xy
            H2_3D_Compute<H2_3D_Tree> compute1
                (*tree[1], all_points, all_points, &charge_xyz[0*nn], 2, &output1_xy[0]);
            // xz
            H2_3D_Compute<H2_3D_Tree> compute2
                (*tree[2], all_points, all_points, &charge_xz[0],     2, &output2_xz[0]);
            // yy
            H2_3D_Compute<H2_3D_Tree> compute3
                (*tree[3], all_points, all_points, &charge_xyz[1*nn], 1, &output3_yy[0]);
            // yz
            H2_3D_Compute<H2_3D_Tree> compute4
                (*tree[4], all_points, all_points, &charge_xyz[1*nn], 2, &output4_yz[0]);
            // zz
            H2_3D_Compute<H2_3D_Tree> compute5
                (*tree[5], all_points, all_points, &charge_xyz[2*nn], 1, &output5_zz[0]);

            // distribute to output
            #pragma omp parallel for
            for (size_t i = 0; i < nn; ++i) {
              *(fmm_res.data()+3*i+0) = output0_xx[i]+output1_xy[nn+i]+output2_xz[nn+i];
              *(fmm_res.data()+3*i+1) = output1_xy[i]+output3_yy[i]+output4_yz[nn+i];
              *(fmm_res.data()+3*i+2) = output2_xz[i]+output4_yz[i]+output5_zz[i];
            }
          }
          gpu_timer.stop();
          double FMM_TIME = gpu_timer.elapsed()/1000;
          spdlog::info("TIME(fmm)={0:.3f}", FMM_TIME);

          if ( USE_GUI ) {
            viewer.data().set_mesh(VV+fmm_res, FF);
          } else {
            igl::writeOFF(string(OUTDIR+"/target_deform.off"), Vq+fmm_res.topRows(Vq.rows()), Fq);  
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
            igl::writeOFF(string(OUTDIR+"/target_deform.off"), Vq+fmm_res.topRows(Vq.rows()), Fq);
            break;
          default:
            return false;
        }
        return true;
      };  

  if ( USE_GUI ) {
    viewer.data().set_mesh(VV, FF);
    viewer.data().show_lines = true;
    viewer.data().show_faces = false;
    viewer.core().align_camera_center(VV, FF);
    viewer.launch();
  } else {
    update();
  }
  
  return 0;
}
