/*!\file	read_sources.cpp
 source file to field, source and charge information from binary files.
*/

#include"old_environment.hpp"

namespace bbfmm3 {
void read_Stress(const std::string& filenameStress, double *stress_dir, const int& N) {
  std::ifstream fin;

    /* Read source */
	fin.open(filenameStress.c_str(),ios::binary);
	
	if (!fin.good()){
          std::cerr << "Failed to open file " << filenameStress << std::endl;
          throw std::runtime_error("Failed to open file!");
	}
    
   
    fin.read((char*) stress_dir, N*sizeof(double));
    fin.close();
}
}
