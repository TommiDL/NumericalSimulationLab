#ifndef __population__
#define __population__

#include <armadillo>
#include <vector>
#include "map.h"
#include "operators.h"
#include "random.h"

using namespace arma;

typedef vector<int> crom;


/*
Contains a set of possible solutions
*/
class population
{
    public:
        population(coordinates &map, int n);

        ~population();

        void print(string prefactor="");
        int get_size() {return _pop_size;};
        vector<crom> get_pop(){return _population;};

        void set_mutation_probability(
            double p_inversion,
            double p_permutation,
            double p_shift 
        );

        void save_population_best();

        void evolve();
    private:

        int _pop_size=10;
        vector<crom> _population;
        selection sel;
        mutation  mut;


        vector<double> mutation_prob = vector<double>(3, 0.1);

        bool check_population();
        bool check_solution(int i);
        ofstream out;
        string best_filename = "best_sol.dat";
        
        //eventually change into L1 or another coordinates cost function



        

};

#endif