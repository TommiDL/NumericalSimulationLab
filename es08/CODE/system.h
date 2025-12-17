#ifndef __System__
#define __System__

#include <iostream>
#include <fstream>
#include <string>
#include "random.h"
//#include "function.h"

using namespace std;

class System
{
    private:

        bool _end=false;

        double mass=1., hbar=1.;

        // actual parameters
        double _mu, _sigma;
        double _x0, _x;


        //proposed step storage values
        double _new_mu, _new_sigma;
        double _new_x;

        double _en0, _energy, _new_energy;        
        
        double _init_temp=0., _temp=0., _beta=0.;
        double _min_temp=0.1, _temp_decr=0.1;   //temperature variational parameters
        double _max_beta=1./_min_temp;
        int _nsteps, eq_steps=0;
        int _nblocks;
        int _naccepted, _nattempts;
        double _delta;


        int _nanneal_accepted, _nanneal_attempts;
        int _anneal_nsteps=200;

        Random _rnd;
        ofstream clog;
        ofstream copt;
        ofstream cen;
        ofstream cenacc;

        double avg, avg2;
        double _block_avg;
        double _error;

        //storage files
        string en_filename="OUTPUT/energy.dat", optimization_filename="OUTPUT/optimization.dat";
        string log_filename="OUTPUT/output.dat";
        string en_acceptance_filename="OUTPUT/en_acceptance.dat";
        string opt_acceptance_filename="OUTPUT/opt_acceptance.dat";
        string wf_filename="OUTPUT/wavefunction.dat";

        //pointers to distribution

        string distribution_type;
        double distr_width;

        //define distribution pointer
        typedef double (Random::*distr) (double, double); 
        //storage distribution variable (default uniform)
        distr annealing_distribution=&Random::centered_Rannyu; 
        

        //annealing schedule
        typedef void (System::*schedule)();
        string annealing_schedule="step";
        schedule temp_decrease=&System::step_cooling;

        double h(double x, double mu, double sigma);
        double V(double x);

        void initialize_rnd();
        void initialize_files();

        double prob(double x, double mu, double sigma);


        //internal loop
        bool metro(double mu, double sigma);
        double error(double acc, double acc2, int blk);

        //external loop
        bool anneal_metro();
        void accept_anneal_move();

        void step_cooling();
        void rate_cooling();
        void step_cooling_beta();
        void rate_cooling_beta();

    public:
        // set methods
        void set_sigma(double sigma);
        void set_mu(double mu);
        void set_pos0(double x);
        void set_energy_filename(string filename);
        void set_optimization_filename(string filename);
        void set_temperature(double temp);
        void set_beta(double beta);
        void set_min_temperature(double mintemp);

        // get methods
        int get_n_annealing_steps();
        double get_temperature();


        void initialize(string inputfile="../INPUT/input.dat");

        void compute_initial_condition();

        void anneal_move();
        void move(double mu, double sigma);
        void equilibration();



        void write_position();

        double local_en(double x, double mu, double sigma);

        double compute_energy(bool paramnew=false, bool writepos=false, bool save_en=false);
        double compute_energy(string enfile);
        void cooling();

        bool terminated();

        void sample_probability(string filename="OUTPUT/wf_prob.dat", int npoints=1000);
};



#endif