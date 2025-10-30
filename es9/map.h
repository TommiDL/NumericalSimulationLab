#ifndef __map__
#define __map__

#include "random.h"
#include <iostream>
#include <armadillo>
#include <fstream>
#include <exception>

using namespace std;
using namespace arma;



/*
Object that contain a numerated list of points 
*/
class coordinates
{
    public:
        // default constructor [create n cities inside a square or on a circle]
        coordinates(int n, string path = "square");
        // read file of coordinates
        coordinates(string filename);
        coordinates(mat coord);

        vec get_coord(int n);

        double dist(int n, int m);
        double dist2(int n, int m);

        double L1 (vector<int> cities);
        double L2 (vector<int> cities);

        void print();
        int get_ncities(){return _ncities;};

        void save_coordinates(string filename);
    private:
        mat _coord;
        int _ndim=2;
        int _ncities;
        void check_index(int n);
        Random _rnd;
        void fill_square(double size=1.);
        void fill_circle(double ray=1.);

        void init_gen(
            string primesfile="Primes",
            string seedfile="seed.in"
        );
        


};

#endif