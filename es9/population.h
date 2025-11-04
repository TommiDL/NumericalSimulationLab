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

        void print(string prefactor="", bool _new=false);
        int get_size() {return _pop_size;};
        vector<crom> get_pop(){return _population;};

        void set_mutation_probability(
            double p_inversion,
            double p_permutation,
            double p_shift 
        );
        void set_crossing_probability(double prob){cross_prob=prob;};


        void evolve();
        double evaluate_sol(int i);


        void compute_costs();

        void save_population_best();
        void sort_population();


    private:

        int _pop_size=10;
        vector<crom> _population;
        vector<crom> _new_population;
        crom _best;

        vector<double> _pop_costs;

        Random _rnd;

        selection sel;
        mutation  mut;
        crossing cross;

        vector<double> mutation_prob = vector<double>(3, 0.1);
        double cross_prob=0.5;


        vector<crom> merge_cost(vector<crom> &pop);

        void subpermutation (vector<crom> &pop, double prob);
        void inversion (int k);
        void permutation (int k);
        void shift (int k);

        void crossing_pop();

        // to change so i can use also prob2
        typedef int (population::*select) ();
        select prob = &population::selection_prob1;

        void compute_prob1();

        int selection_prob1();
        int selection_prob2(double p = 2.);



        bool check_population(bool _new=false);
        bool check_solution(int i, bool _new);
        ofstream out;
        string best_filename = "best_sol.dat";
        
        //eventually change into L1 or another coordinates cost function



        

};

#endif