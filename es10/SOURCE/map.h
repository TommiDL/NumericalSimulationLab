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
        coordinates();
        // default constructor [create n cities inside a square or on a circle]
        coordinates(int n, string path = "square");
        // read file of coordinates
        coordinates(string filename);
        coordinates(mat coord);

        void initialize(string inputfile="input.dat");

        /*return the coordinates corresponding to as a vec*/
        vec get_coord(int n);

        /*return the eucledian distance between the cities n and m*/
        double dist(int n, int m);
        /*return the quadratic eucledian distance between the cities n and m*/
        double dist2(int n, int m);

        /*return the eucledian lenght of the ...*/
        double L1 (vector<int> cities);
        double L2 (vector<int> cities);

        void print();
        int get_ncities(){return _ncities;};

        void save_coordinates(string filename);
    private:
        mat _coord;
        int _ndim=2;
        int _ncities=34;
        void check_index(int n);
        Random _rnd;
        void fill_square(double size=1.);
        void fill_circle(double ray=1.);

        void init_gen(
            string primesfile="Primes",
            string seedfile="seed.in"
        );
        
        void read_from_file(string filename);


};

#endif