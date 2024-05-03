/*!\file read_sources.hpp
 read sources from binary files
*/
#ifndef __old_read_sources_hpp__
#define __old_read_sources_hpp__

#include"old_environment.hpp"
#include"old_bbfmm.h"

namespace bbfmm3 {

void read_Sources(const std::string& filenameField, vector3 *field, const int& Nf, const std::string& filenameSource, vector3 *source, const int& Ns, const std::string& filenameCharge, double *q, const int& m, const doft& dof);

}
#endif //(__read_sources_hpp__)
