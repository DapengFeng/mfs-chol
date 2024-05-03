/*!\file read_metadata.cpp
 */

#include"old_read_metadata.hpp"


namespace bbfmm3 {
void read_Metadata(const std::string& filenameMetadata,double& L, int& n, doft& dof, int& Ns, int& Nf, int& m, int& level) {
  std::ifstream fin;
	fin.open(filenameMetadata.c_str());
	if (!fin.good()){
          std::cerr << "Failed to open file " << filenameMetadata << std::endl;
          throw std::runtime_error("Failed to open file!");
	}
        std::string line;
    getline(fin,line);
    line.erase(remove(line.begin(), line.end(), ' '),
               line.end());
    std::stringstream ss;
    ss << line;
    char comma;
    ss >> L >> comma >> n >> comma >> dof.s >> comma >> dof.f >> comma >>
    Ns >> comma >> Nf >> comma >> m >> comma >> level;
    fin.close();
}

}
