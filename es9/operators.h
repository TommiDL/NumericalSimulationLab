#ifndef __operators__
#define __operators__

#include "map.h"
#include "random.h"
#include <utility>
#include <armadillo>
#include <vector>
#include "function.h"

using namespace std;
using namespace arma;
// define cost variable type (pointer to function)
typedef double (coordinates::*cost) (vector<int>);

class selection
{
    public:
        selection();
        selection(coordinates &map);
        selection(int n_points, string path="square");
        selection(string filename);

        ~selection();

        void set_metric(string norm = "L1");
        double evaluate_sol(crom sol);
        
        int get_ncities(){return this->_map->get_ncities();};

    private:

        // members
        coordinates *_map;
        bool existing_coord=true; // to eventually deallocate pointer

        cost C=&coordinates::L2;  //pointer to function


};

class mutation
{
    public:
        mutation();
        ~mutation();

        crom permutation(crom sol);
        crom shift(crom sol, int n, int m);
        crom subpermutation(crom sol, int m);
        crom inversion(crom sol, int m);

    private:
        Random _rnd;

        crom swap(crom sol, int start1, 
            int start2, int size);

};



class crossing
{
    public:
        crossing();
        //pair<crom, crom> cross(crom parent1, crom parent2);
        pair<crom, crom> cross(
            crom parent1, crom parent2, int crosspoint
        );
        
        pair<crom, crom> cross(
            crom parent1, crom parent2
        );



    private:
        Random _rnd;


};


#endif