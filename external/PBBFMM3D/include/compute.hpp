
/*! \file compute.hpp
 */
#ifndef __compute_hpp__
#define __compute_hpp__
#include <vector>
#include "bbfmm.h"
#include "environment.hpp" 
#include "kernel_Types.hpp"
#include <omp.h>
#include <unistd.h>
#include <limits.h>

#define FFTW_FLAG       FFTW_ESTIMATE // option for fftw plan type

template <typename T>
class H2_3D_Compute {
public:
    H2_3D_Compute(T& FMMTree, const std::vector<vector3>& target, const std::vector<vector3>&source, std::vector<double>& charge, int nCols, std::vector<double>& output);
    H2_3D_Compute(T& FMMTree, const std::vector<vector3>& target, const std::vector<vector3>&source, double *charge, int nCols, double *output);
  
    T* FMMTree;
    vector3 * target;
    vector3 * source;
    std::vector<vector3> mtarget;
    std::vector<vector3> msource;
    int Ns;
    int Nf;
    double* charge;
    int nCols;
    int tree_level;
    nodeT** indexToLeafPointer;
    std::vector<nodeT*> cellPointers;
    void FMMDistribute(nodeT **A, vector3 *target, vector3 *source, int Nf,
                       int Ns, int tree_level);
    void FMMCompute(nodeT **A, vector3 *target, vector3 *source, double *charge,
                    double *K, double *U, double *VT, double *Tkz, int *Ktable,
                    double *Kweights, double *Cweights, double homogen,
                    doft *cutoff, int n, doft *dof,double*output, int use_chebyshev);
    void UpwardPass(nodeT **A, vector3 *source, double *weight, double *Cweights, double *Tkz, double *VT, 
        double *Kweights, doft *cutoff, int n, doft *dof,  int homogen, int curTreeLevel, int use_chebyshev);
    void FarFieldInteractions(double *E, int *Ktable, double *U, 
            double *VT, double *Kweights, int n, doft dof,
            doft cutoff, double homogen, int
            use_chebyshev);
    void NearFieldInteractions(vector3 *target, vector3 *source,
                  double *weight, int n, double *Tkz, doft *dof, double *phi, nodeT** indexToLeafPointer, int use_chebyshev);
    void DownwardPass(nodeT **A, vector3 *target, vector3 *source,
                      double *Cweights, double *Tkz, double *weight, doft *cutoff,
                      int n, doft *dof, double *output, int use_chebyshev);
    void EvaluateField(vector3* target, vector3* source, int Nf,int Ns, doft *dof, double* Kcell);
    void EvaluateField_self(vector3* target, int Nf, doft *dof, double* Kcell);

    void Local2Local(int n, double *r, double *F, double *Fchild, doft *dof, double *Cweights, int use_chebyshev);
    void Moment2Local(int n, double *R, double *cell_mpCoeff, double *FFCoeff,
                                    double *E, int *Ktable, doft dof, doft cutoff, int use_chebyshev);
    void Moment2Moment(int n, double *r, double *Schild, double *SS, doft *dof, double *Cweights);
    void InteractionList(nodeT **A, int levels);
    void DistributeFieldPoints(nodeT **A, vector3 *target, int *targetlist,
                               int levels);
    void FrequencyProduct(int N, double *Afre, double *xfre, double *res);
    void FreeNode(nodeT *A);

    ~H2_3D_Compute();
};


#endif //(__compute_hpp__)
