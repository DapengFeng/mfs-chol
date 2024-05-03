#include <iostream>
#include <fstream>
#include <Eigen/Sparse>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "src/img2grid.h"
#include "src/types.h"
#include "src/preprocess.h"
#include "src/timer.h"
#include "src/gputimer.h"
#include "src/kl_chol.h"
#include "src/ptree.h"

using namespace cv;
using namespace klchol;
using namespace std;
using namespace Eigen;

static void extend_detected(const Mat &orig, Mat &ext)
{
  orig.copyTo(ext);
  ext = Scalar::all(0);

  #pragma omp parallel for
  for (int iter = 0; iter < orig.rows*orig.cols; ++iter) {
    const int i = iter%orig.rows, j = iter/orig.rows;
    if ( orig.at<uchar>(i, j) > 0 ) {
      ext.at<uchar>(i, j) = 255;
      if ( i+1 < orig.rows ) {
        ext.at<uchar>(i+1, j) = 255;
      }
      if ( i-1 >= 0 ) {
        ext.at<uchar>(i-1, j) = 255;
      }
      if ( j+1 < orig.cols ) {
        ext.at<uchar>(i, j+1) = 255;
      }
      if ( j-1 >= 0 ) {
        ext.at<uchar>(i, j-1) = 255;
      }
      if ( i-1 >= 0 && j-1 >= 0 ) {
        ext.at<uchar>(i-1, j-1) = 255;
      }
      if ( i-1 >= 0 && j+1 < orig.cols ) {
        ext.at<uchar>(i-1, j+1) = 255;
      }
      if ( i+1 < orig.rows && j-1 >= 0 ) {
        ext.at<uchar>(i+1, j-1) = 255;
      }
      if ( i+1 < orig.rows && j+1 < orig.cols ) {
        ext.at<uchar>(i+1, j+1) = 255;
      }
    }
  }
}

static string get_file_name(const string &full_path) {
  std::string base_filename = full_path.substr(full_path.find_last_of("/ ") + 1);
  std::string::size_type const p(base_filename.find_last_of('.'));
  std::string file_without_extension = base_filename.substr(0, p);
  return file_without_extension;      
}

template <class Cont> 
static void write_residual(const string &filename, const Cont &resd, const double dt)
{
  ofstream ofs(filename);
  ofs << "iter, time, resd" << endl;
  for (size_t i = 0; i < resd.size(); ++i) {
    ofs << i << "," << i*dt << "," << resd[i] << endl;
  }
  ofs.close();
}

template <class Cont>
static void write_residual(const string &filename, const Cont &resd,
                           const double dt, const double time_before_solve)
{
  ofstream ofs(filename);
  ofs << "iter, time, resd" << endl;
  ofs << 0 << "," << time_before_solve << ",1.0" << endl;
  for (size_t i = 1; i <= resd.size(); ++i) {
    ofs << i << "," << time_before_solve+(i-1)*dt << "," << resd[i-1] << endl;
  }
  ofs.close();  
}

int main(int argc, char **argv)
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  Mat src, src_gray;
  Mat dst, detected_edges, detected_edges_ext;

  const int lowThreshold = pt.get<int>("low_threshold.value");
  const std::string outdir = pt.get<string>("outdir.value");
  const int ratio = 3;
  const int kernel_size = 3;
  const string img_file = pt.get<string>("input.value");
  const string img_name = get_file_name(img_file);

  const bool show_img = pt.get<bool>("show_img.value", false);

  src = imread(samples::findFile(img_file), IMREAD_COLOR );
  if ( src.empty() )
  {
    std::cout << "Could not open or find the image!\n" << std::endl;
    return -1;
  }

  // create windows
  //  namedWindow("orig", WINDOW_AUTOSIZE);
  namedWindow("canny", WINDOW_AUTOSIZE);
  namedWindow("final", WINDOW_AUTOSIZE);

  // color to gray
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  //  imshow("orig", src);
  
  // canny
  blur(src_gray, detected_edges, Size(3,3));
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
  extend_detected(detected_edges, detected_edges_ext);

  // image with canny as mask
  dst.create(src.size(), src.type());
  dst = Scalar::all(0);
  src.copyTo(dst, detected_edges_ext);
  if ( show_img ) {
    imshow("canny", dst);
    waitKey(0);
  }
  {
    char outfile[256];
    sprintf(outfile, "%s/%s-canny.png", outdir.c_str(), img_name.c_str());
    imwrite(outfile, dst);
  }

  // convert image to point cloud
  klchol::img_grid_converter<double> conv(dst, 1.0);
  // extract colored pixels
  klchol::RmMatF_t V, Cd, Vq;
  Eigen::VectorXi colored_pix;
  {
    conv.select_colored_pixels(V, Cd, colored_pix);
    conv.to_grid(Vq);
    spdlog::info("train size={}, prediction size={}", V.rows(), Vq.rows());
  }

  // PBBFMM data storage
  std::vector<vector3> all_pix(Vq.rows());
  for (size_t i = 0; i < all_pix.size(); ++i) {
    all_pix[i].x = Vq(i, 0);
    all_pix[i].y = Vq(i, 1);
    all_pix[i].z = Vq(i, 2);      
  }  
  
  GpuTimer gpu_timer;

  // preprocessing
  fps_sampler fps(V);
  Eigen::PermutationMatrix<-1, -1> P_tr;
  gpu_timer.start();
  {
    fps.compute('F');
    fps.debug();
    fps.reorder_geometry(V);
    P_tr = fps.P();
    Cd = P_tr*Cd;
  }
  gpu_timer.stop();
  const double FPS_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(FPS)={0:.3f} s", FPS_TIME);
  
  // solver configuration
  klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_2D;
  if ( pt.get<bool>("use_screen_kernel.value", false) ) {
    cout << "\tuse screened Poisson kernel" << endl;
    KTYPE = klchol::KERNEL_TYPE::BILAP_REG_3D;
  }
  const int NUM_SEC = pt.get<int>("num_sec.value");
  const double EPS = pt.get<double>("eps.value");
  const double RHO = pt.get<double>("rho.value");
  const double LAMBDA = pt.get<double>("lambda.value");
  const int MAX_SUPERNODE_SIZE = pt.get<int>("max_supernode_size.value");
  
  // solver
  const size_t DIM = 1;
  const size_t N = V.rows();
  std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON>> super_solver;
  super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON>(N, DIM, NUM_SEC));  
  super_solver->set_source_points(N, V.data());
  super_solver->set_target_points(Vq.rows(), Vq.data());

  VectorXd PARAM = VectorXd::Zero(8);
  PARAM[0] = EPS;
  PARAM[1] = LAMBDA;
  PARAM[2] = RHO;
  PARAM[7] = KTYPE;
  super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
  spdlog::info("rho={0:.3f}, eps={1:.6f}, lambda={2:.6f}", RHO, EPS, LAMBDA);

  Eigen::SparseMatrix<double> PATT, SUP_PATT;
  VectorXi sup_ptr, sup_ind, sup_parent;
  std::vector<size_t> GROUP{0, N};
  gpu_timer.start();
  {
    fps.simpl_sparsity(RHO, DIM, PATT);
    fps.aggregate(DIM, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
    fps.super_sparsity(DIM, PATT, sup_parent, SUP_PATT);    
  }
  gpu_timer.stop();
  const double PATT_BUILD_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(patt_build)={0:.3f} s", PATT_BUILD_TIME);

  gpu_timer.start();
  {
    super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
    super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
  }
  gpu_timer.stop();
  const double PATT_CPY_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(patt_copy)={0:.3f} s", PATT_CPY_TIME);

  gpu_timer.start();
  {
    super_solver->assemble();
  }
  gpu_timer.stop();
  const double ASM_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(assemble)={0:.3f} s", ASM_TIME);
  
  gpu_timer.start();
  {              
    super_solver->factorize();
  }
  gpu_timer.stop();
  const double FAC_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(factorization)={0:.3f} s", FAC_TIME);

  spdlog::info("TIME(PRECOMPUTE)={0:.3f} s", FPS_TIME+PATT_BUILD_TIME+PATT_CPY_TIME);
  spdlog::info("TIME(COMPUTE)={0:.3f} s", ASM_TIME+FAC_TIME);
  const double TIME_BEFORE_SOLVE = FPS_TIME+PATT_BUILD_TIME+PATT_CPY_TIME+ASM_TIME+FAC_TIME;

  FMM_parameters fmm_config;
  {
    fmm_config.L                   = 1.0;
    fmm_config.tree_level          = V.rows() > 180000 ? 6 : 5; // emperical
    fmm_config.interpolation_order = pt.get<int>("fmm_order.value", 4);
    fmm_config.eps                 = pt.get<double>("fmm_eps.value", 1e-5);
    fmm_config.use_chebyshev       = 1;
    fmm_config.reg_eps             = EPS;
    fmm_config.K                   = LAMBDA;
  }
  
  // boundary impulses
  const int pcg_iters = pt.get<int>("pcg_iters.value");
  vector<double> pcg_resd;
  RmMatF_t res = RmMatF_t::Zero(N, 3);
  VectorXd color(N), y(N);
  std::pair<int, double> pcg_ret;
  gpu_timer.start();
  {
#if 1
    for (int k = 0; k < 3; ++k) {
      std::cout << "--- solve " << k << " th rhs ---" << std::endl;
      color = Cd.col(k);
      y.setZero();
      pcg_ret = super_solver->pcg(color.data(), y.data(), true, pcg_iters, 1e-6, &fmm_config, &pcg_resd);
      spdlog::info("pcg ret={0:d}, {1:.6f}", pcg_ret.first, pcg_ret.second);
      res.col(k) = y;
    }
#else
    super_solver->thread_safe_pcg_precomp(&fmm_config);

    #pragma omp parallel for num_threads(3)
    for (int c = 0; c < 3; ++c) {
      VectorXd color_val = Cd.col(c), y = VectorXd::Zero(N);
      auto pcg_ret = super_solver->thread_safe_pcg(color_val.data(), y.data(), true, &fmm_config, pcg_iters, 1e-4);
      spdlog::info("pcg ret={0:d}, {1:.6f}", pcg_ret.first, pcg_ret.second);
      res.col(c) = y;
    }
#endif
  }
  gpu_timer.stop();
  double SLV_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(pcg)={0:.3f} s", SLV_TIME);
  char outfile[256];
  sprintf(outfile, "%s/%s-pcg-%d.txt", outdir.c_str(), img_name.c_str(), pcg_iters);
  write_residual(outfile, pcg_resd, 0.33333*SLV_TIME/(pcg_resd.size()-1));
  sprintf(outfile, "%s/%s-pcg-with-precomp-%d.txt", outdir.c_str(), img_name.c_str(), pcg_iters);
  write_residual(outfile, pcg_resd, 0.33333*SLV_TIME/(pcg_resd.size()-1), TIME_BEFORE_SOLVE);

  const int cg_iters = pt.get<int>("cg_iters.value");
  if ( cg_iters > 0 ) {
    spdlog::info("cg_iters={}", cg_iters);
    
    gpu_timer.start();
    {
      int k = 2;
      std::cout << "--- solve " << k << " th rhs ---" << std::endl;      
      color = Cd.col(k);
      y.setZero();
      auto ret_wo = super_solver->pcg(color.data(), y.data(), false, cg_iters, 1e-4, &fmm_config, &pcg_resd);
      spdlog::info("normal pcg ret={0:d}, {1:.6f}", ret_wo.first, ret_wo.second);
    }
    gpu_timer.stop();
    double NPCG_TIME = gpu_timer.elapsed()/1000;
    spdlog::info("TIME(NORMAL_PCG)={0:.3f} s", NPCG_TIME);
    
    char outfile[256];
    sprintf(outfile, "%s/%s-cg-%d.txt", outdir.c_str(), img_name.c_str(), cg_iters);
    write_residual(outfile, pcg_resd, NPCG_TIME/(pcg_resd.size()-1));
    sprintf(outfile, "%s/%s-cg-with-precomp-%d.txt", outdir.c_str(), img_name.c_str(), cg_iters);
    write_residual(outfile, pcg_resd, NPCG_TIME/(pcg_resd.size()-1), 0.0);
  }

  if ( pt.get<bool>("debug_fmm_eval.value") ) {
    // debug fmm approximation    
    VectorXd Kx(N);
    std::vector<double> Kx_fmm(N);
    super_solver->evalKx(Kx.data());
    super_solver->evalKx_fmm(Kx_fmm, fmm_config);      
    std::cout << "[Debug] fmm error=" << sqrt((Map<VectorXd>(&Kx_fmm[0], N)-Kx).squaredNorm()/Kx.squaredNorm()) << std::endl;
  }

  // NEW PBBFMM
  CmMatF_t fmm_res = CmMatF_t::Zero(Vq.rows(), 3), fmm_color = fmm_res;
  fmm_res(colored_pix.array(), Eigen::all) = P_tr.inverse()*res;
  gpu_timer.start();
  {
    fmm_config.tree_level = 6;
    std::vector<std::shared_ptr<H2_3D_Tree>> trees;    
    klchol::create_fmm_tree(trees, KTYPE, fmm_config);
    H2_3D_Compute<H2_3D_Tree> compute
        (*trees[0], all_pix, all_pix, fmm_res.data(), 3, fmm_color.data());
  }
  gpu_timer.stop();
  double FMM_PRED_TIME = gpu_timer.elapsed()/1000;
  spdlog::info("TIME(fmm_extrap)={0:.3f} s", FMM_PRED_TIME);

  double MIN_Cd = 0, MAX_Cd = 1.0;  
  if ( pt.get<bool>("normalize_by_maxmin.value", false) ) {
    MIN_Cd = fmm_color.minCoeff();
    MAX_Cd = fmm_color.maxCoeff();
  }
  const auto &to_255 =
      [&](const double x) -> int {
        return std::max(0.0, std::min(1.0, (x-MIN_Cd)/(MAX_Cd-MIN_Cd)))*255;
      };
  
  // cuda based extrapolation
  RmMatF_t cuda_color = RmMatF_t::Zero(Vq.rows(), 3);
  if ( pt.get<bool>("test_cuda_extrap.value") ) {
    gpu_timer.start();
    {
      super_solver->predict(3, res.data(), res.rows(), cuda_color.data(), cuda_color.rows());
    }
    gpu_timer.stop();   
    spdlog::info("TIME(cuda_extrap)={0:.3f} s", gpu_timer.elapsed()/1000);
    spdlog::info("fmm extrap error={0:.6f}", (fmm_color-cuda_color).norm()/cuda_color.norm());

    // interpolate final images
    Mat cuda_img = cv::Mat::ones(src.rows, src.cols, CV_8UC3);
    #pragma omp parallel for
    for (size_t iter = 0; iter < cuda_img.rows*cuda_img.cols; ++iter) {
      const int i = iter/cuda_img.cols, j = iter%cuda_img.cols;
      cuda_img.at<cv::Vec3b>(i, j) = cv::Vec3b(
          to_255(cuda_color(iter, 0)),
          to_255(cuda_color(iter, 1)),
          to_255(cuda_color(iter, 2)));
    }
    {
      char outfile[256];
      sprintf(outfile, "%s/%s-cuda-pcg-%d.png", outdir.c_str(), img_name.c_str(), pcg_iters);
      imwrite(outfile, cuda_img);
    }    
  }

  // OLD BBFMM
  if ( pt.get<bool>("test_old_fmm.value") ) {
    fmm_config.tree_level = 6;
    
    CmMatF_t colm_res = res, old_color = CmMatF_t::Zero(Vq.rows(), 3);
    gpu_timer.start();
    {
      super_solver->predict_fmm(
          3, colm_res.data(), colm_res.rows(),
          old_color.data(), old_color.rows(), fmm_config);
    }
    gpu_timer.stop();
    spdlog::info("TIME(old_fmm_extrap)={0:.3f} s", gpu_timer.elapsed()/1000);
    spdlog::info("(old-new)/new={0:.6f}", (old_color-fmm_color).norm()/fmm_color.norm());
  }
    
  // interpolate final images
  Mat float_img = cv::Mat::ones(src.rows, src.cols, CV_8UC3);
  #pragma omp parallel for
  for (size_t iter = 0; iter < float_img.rows*float_img.cols; ++iter) {
    const int i = iter/float_img.cols, j = iter%float_img.cols;
    float_img.at<cv::Vec3b>(i, j) = cv::Vec3b(
        to_255(fmm_color(iter, 0)),
        to_255(fmm_color(iter, 1)),
        to_255(fmm_color(iter, 2)));
  }
  cv::Mat normalizedImage = float_img;
  if ( show_img ) {
    imshow("final", normalizedImage);
  }
  {
    char outfile[256];
    sprintf(outfile, "%s/%s-final-pcg-%d.png", outdir.c_str(), img_name.c_str(), pcg_iters);
    imwrite(outfile, normalizedImage);
  }

  if ( show_img ) {
    waitKey(0);
  }
  std::cout << "# done!" << std::endl;
  return 0;
}
