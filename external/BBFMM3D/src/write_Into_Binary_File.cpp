
/*!\file	write_Into_Binary_File.cpp
*/

#include"old_write_Into_Binary_File.hpp"

using namespace std;

namespace bbfmm3 {
void write_Into_Binary_File(const string& filename, double* data, int numOfElems) {
    
    ofstream outdata;
	outdata.open(filename.c_str(),ios::binary);
    
    outdata.write((char *)data, numOfElems*sizeof(double));
	
	outdata.close();
}
}
