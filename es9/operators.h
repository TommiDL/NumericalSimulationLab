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
        vector<crom> sort_population(vector<crom> &pop);
        double evaluate_sol(crom sol);

        void selection_prob1(vector<crom> &pop);
        void selection_prob2(vector<crom> &pop, double p = 2.);
        void save_population_best(vector<crom> &pop, string filename="best_sol.dat");


    private:

        // members
        coordinates *_map;
        bool existing_coord=true;

        vector<double> _pop_costs;
        crom _best;
        Random _rnd;

        cost C=&coordinates::L2;  //pointer to function

        void compute_costs(vector<crom> &pop);

        void compute_prob1(vector<crom> &pop);

        void init_gen(
            string primesfile="Primes", int primerow=2, 
            string seedfile="seed.in"
        );

};

class mutation
{
    public:
        mutation();
        ~mutation();

        crom permutation(crom sol);
        void permutation (vector<crom> &pop, double prob);

        crom shift(crom sol, int n, int m);
        void shift (vector<crom> &pop, double prob);

        crom subpermutation(crom sol, int m);
        void subpermutation (vector<crom> &pop, double prob);

        crom inversion(crom sol, int m);
        void inversion (vector<crom> &pop, double prob);

    private:
        Random _rnd;

        crom swap(crom sol, int start1, 
            int start2, int size);

        void init_gen(
            string primesfile="Primes", int primerow=3, 
            string seedfile="seed.in"
        );
};


/*
class crossing
{
    public:
        crossing();

        pair<vec, vec> cross(const vec &parent1, const vec &parent2, int cpoint);


        
    private:
        int _dim;



};


class mutation
{};
*/
#endif