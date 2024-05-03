/*! \file	read_metadata.hpp
*/
#ifndef __old_read_metadata_hpp__
#define __old_read_metadata_hpp__

#include"old_environment.hpp"
#include"old_bbfmm.h"


// using namespace std;

namespace bbfmm3 {
void read_Metadata(const std::string& filenameMetadata,double& L, int& n, doft& dof, int& Ns, int& Nf, int& m, int& level);
}

#endif //(__read_metadata_hpp__)
