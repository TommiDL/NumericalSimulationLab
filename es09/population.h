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
        population(coordinates &map);
        

        ~population();

        void initialize(string inputfile="input.dat");
        /*Print population

        @param string prefactor: string to print before the single population's member
        @param bool _new: 
            - true : print the new proposed population
            - false: print the current population*/
        void print(string prefactor="", bool _new=false);
        
        
        int get_size() {return _pop_size;};
        
        /*set the probability of executing mutation operators on the population's members*/
        void set_mutation_probability(
            double p_inversion,
            double p_permutation,
            double p_shift,
            double p_subpermutation
        );

        /*set probability of crossing*/
        void set_crossing_probability(double prob){cross_prob=prob;};

        /*
        Evolve population using crossing and mutation on members.

        Start from the actual population and generate a new one selecting member using 
        a probability monotone function of fitness and applying to this selcted members 
        the crossing and mutation operators with the selected probabilities  
        */
        void evolve();

        /*
        Return cost of i-th member of the population
        To execute only after compute_prob1  
        */
        double evaluate_sol(int i);


        /*save population best solution on file*/
        void save_population_best(); 

        /*merge sort based on the coordinates cost*/
        void sort_population();

        /*set path to save population best member*/
        void set_best_filename(string filename);
        /*set path to save average of population's best half*/        
        void set_avg_filename(string filename);

        /*set path to save average of population's best half*/        
        void set_setup_filename(string filename);

        void set_nsteps(int nsteps);

        /*save on file average of population's best half*/        
        void save_avg_cost();
        void save_best_half(string filename, int iteration);

        void initialize_files();

        void print_sol(int k, bool _new=false);

        //void initialize_newpop();
        void print_setup();


        void perform_evolution();

    private:

        int _pop_size=100, _evolution_steps=100, _ncities;
        vector<crom> _population;       //actual population
        vector<crom> _new_population;   //new proposed population
        crom _best;                     //store best sol of actual population

        vector<double> _pop_costs;      //store the cost of population's members
        bool cost_evaluated=false, sorted=false;        
        double _avg_cost=0.;
        
        Random _rnd;

        selection sel;
        mutation  mut;
        crossing cross;

        vector<double> mutation_prob = vector<double>(4, 0.1);
        double cross_prob=0.5;

        void generate_initial_population(int n);
        /*Compute cost (total distance) of a vector of cromosomes*/
        void compute_costs();

        /*merge sort based on the coordinates cost*/
        vector<crom> merge_cost(vector<crom> &pop);


        void inversion (int k);
        void permutation (int k);
        void shift (int k);
        void subpermutation (int k);

        void crossing_pop();

        // to change so i can use also prob2
        typedef int (population::*select) ();
        select prob = &population::selection_prob1;


        /*Compute probabilities of the solutions and store it 
        in the _pop_costs vector using the distance between cities*/
        void compute_prob1();

        /*select an integer following probability 1 distribution*/
        int selection_prob1();
        /*select an integer following probability 2 distribution*/
        int selection_prob2(double p = 2.);

        
        /*
        Iterate over every member of the population to see 
        if there is a non-valid solution
        @param bool _new 
            - true : check new proposed population
            - false: check current population
        */
        bool check_population(bool _new=false);

        

        /*
        Check solution at index i

        @param int i: index of the population member to check
        @param bool _new 
            - true : check in the new proposed population
            - false: check in the current population 
        */
        bool check_solution(int i, bool _new);
        
        ofstream out;
        string best_filename = "OUTPUT/best_sol.dat";  
        string avg_filename  = "OUTPUT/avg_cost.dat";
        string setup_filename = "OUTPUT/setup.dat";    

};

#endif