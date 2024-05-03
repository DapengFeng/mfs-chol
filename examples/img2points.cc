#include <iostream>
#include <fstream>
#include <Eigen/Sparse>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "src/img2grid.h"
#include "src/types.h"
// #include "src/preprocess.h"
// #include "src/timer.h"
// #include "src/gputimer.h"
// #include "src/kl_chol.h"
#include "src/ptree.h"

using namespace cv;
using namespace klchol;
using namespace std;
using namespace Eigen;

static string get_file_name(const string &full_path) {
  std::string base_filename = full_path.substr(full_path.find_last_of("/ ") + 1);
  std::string::size_type const p(base_filename.find_last_of('.'));
  std::string file_without_extension = base_filename.substr(0, p);
  return file_without_extension;      
}

int main(int argc, char **argv)
{
  boost::property_tree::ptree pt;
  mschol::read_cmdline(argc, argv, pt);
  
  Mat src, src_gray;
  Mat dst, detected_edges, detected_edges_ext;

  const int lowThreshold = pt.get<int>("low_threshold.value", 20);
  //  const std::string outdir = pt.get<string>("outdir.value");
  const int ratio = 3;
  const int kernel_size = 3;
  const string img_file = pt.get<string>("input.value");
  const string img_name = get_file_name(img_file);

  src = imread(samples::findFile(img_file), IMREAD_COLOR );
  if ( src.empty() )
  {
    std::cout << "Could not open or find the image!\n" << std::endl;
    return -1;
  }

  // create windows
  namedWindow("orig", WINDOW_AUTOSIZE);
  namedWindow("canny", WINDOW_AUTOSIZE);

  // color to gray
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  imshow("orig", src_gray);
  std::cout << "src_gray type=" << src_gray.type() << std::endl;
  {
    klchol::img_grid_converter<double> conv(src_gray, 1.0);
    klchol::RmMatF_t V, Cd;
    Eigen::VectorXi colored_pix;
    {
      conv.select_black_pixels(V, colored_pix);
    }
    cout << V.rows() << endl;

    std::ofstream ofs(pt.get<string>("output_black.value"));
    for (size_t i = 0; i < V.rows(); ++i) {
      ofs << "v " << V(i, 0)+1 << " " << V(i, 1)+1 << " " << V(i, 2)+1 << endl;
    }
    ofs.close();
  }
  
  // canny
  blur(src_gray, detected_edges, Size(3,3));
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
  imshow("canny", detected_edges);
  std::cout << "canny type=" << detected_edges.type() << std::endl;
  {
    klchol::img_grid_converter<double> conv(detected_edges, 1.0);
    klchol::RmMatF_t V, Cd;
    Eigen::VectorXi colored_pix;
    {
      conv.select_colored_pixels(V, Cd, colored_pix);
    }
    cout << V.rows() << endl;

    std::ofstream ofs(pt.get<string>("output.value"));
    for (size_t i = 0; i < V.rows(); ++i) {
      ofs << "v " << V(i, 0)+1 << " " << V(i, 1)+1 << " " << V(i, 2)+1 << endl;
    }
    ofs.close();
  }

  // image with canny as mask
  // dst.create(src.size(), src.type());
  // dst = Scalar::all(0);
  // src.copyTo(dst, detected_edges_ext);
  // {
  //   char outfile[256];
  //   sprintf(outfile, "%s/%s-canny.png", outdir.c_str(), img_name.c_str());
  //   imwrite(outfile, dst);
  // }
  
  waitKey(0);  
  std::cout << "# done!" << std::endl;
  return 0;
}
