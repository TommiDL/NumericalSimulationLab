#include "system.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iterator>
#include <ostream>
#include <string>


//private
double System :: h(double x, double mu, double sigma)
{
  double c = - pow(this->hbar, 2) / (2* this->mass); 
  double h1 = (x*x + mu*mu)/pow(sigma,2)-1; 
  double ep = exp(-pow(x+mu,2)/(2*pow(sigma, 2)));
  double em = exp(-pow(x-mu,2)/(2*pow(sigma, 2)));
  double h2 = (2*x*mu/pow(sigma,2))* (ep-em)/ (ep+em);

  double h= c* (h1 + h2)/pow(sigma,2);

  return h;
}

double System :: V(double x)
{
    return pow(x, 4) -5.*pow(x,2)/2.;
}

void System::initialize_rnd()
{

  // INIT RANDOM GENERATOR
  int p1, p2; // Read from ../INPUT/Primes a pair of numbers to be used to initialize the RNG
  ifstream Primes("../INPUT/Primes");
  Primes >> p1 >> p2 ;
  Primes.close();
  int seed[4]; // Read the seed of the RNG
  ifstream Seed("../INPUT/seed.in");
  Seed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
  this->_rnd.SetRandom(seed,p1,p2);
}

void System::initialize_files()
{
    clog.open(log_filename);
    if(!clog.is_open()) 
    {
        cerr << "ERROR: Unable to open " << log_filename<< endl;
    }


    copt.open(optimization_filename);
    if(!copt.is_open())     {
        cerr << "ERROR: Unable to initialize "<< optimization_filename << endl;
        clog << "ERROR: Unable to initialize "<< optimization_filename << endl;

        exit(-1);
    }
    copt<< "TEMP, STEP, MU, SIGMA, BLOCK, EN, ERR" << endl;



    ofstream ceap(en_acceptance_filename);
    if(!ceap.is_open()) 
    {
        cerr << "ERROR: Unable to initialize "<< en_acceptance_filename << endl;
        clog << "ERROR: Unable to initialize "<< en_acceptance_filename << endl;

        exit(-1);
    }
    ceap << "BLOCK, ACCEPTANCE" << endl;
    ceap.close();

    ofstream coap(opt_acceptance_filename);
    if(!coap.is_open()) 
    {
        cerr << "ERROR: Unable to initialize "<< opt_acceptance_filename << endl;
        clog << "ERROR: Unable to initialize "<< opt_acceptance_filename << endl;

        exit(-1);
    }
    coap << "BLOCK, ACCEPTANCE" << endl;
    coap.close();

    ofstream cpos(wf_filename);
    if(!cpos.is_open()) 
    {
        cerr << "ERROR: Unable to initialize "<< wf_filename << endl;
        clog << "ERROR: Unable to initialize "<< wf_filename << endl;
    }
    cpos<<"pos"<<endl;
    cpos.close();




}

//public
double System::local_en(double x, double mu, double sigma)
{
    return h(x, mu, sigma)+V(x);
}



void System::move(double mu, double sigma)
{
    _new_x = _x +_rnd.Rannyu(-1., 1.)*_delta;
    
    if(metro(mu, sigma))
    {
        _x = _new_x;        
        _naccepted+=1;
    }
    _nattempts+=1;

    return;

}

void System::equilibration()
{
    for(int i=0; i< eq_steps; i++)
    {
        this->move(_mu, _sigma); 
        // anche qui mettere possibilita' di cambiare????
    }


    clog<<"Executed "<<eq_steps<<" equilibration steps"<<endl;
    return;
}

bool System::metro(double mu, double sigma)
{
    bool decision = false;
    double acc;

    acc=prob(_new_x, mu, sigma)/prob(_x, mu, sigma);
    if(_rnd.Rannyu()<acc) decision = true;
    return decision;
}

double System::prob(double x, double mu, double sigma)
{
    return pow(
    exp(-pow(x - mu, 2) / (2*pow(sigma, 2))) 
    + exp(-pow(x + mu, 2) / (2*pow(sigma, 2))),
    2 
  );

}

double System :: error(double acc, double acc2, int blk){
  if(blk <= 1) return 0.0;
  else return sqrt( fabs(acc2/double(blk) - pow( acc/double(blk) ,2) )/double(blk) );
}


double System::compute_energy(bool paramnew, bool writepos, bool save_en)
{

    this->equilibration();

    avg=0;
    avg2=0;
    //_error=0;

    _naccepted=0;
    _naccepted=0;

    double m=(paramnew)?_new_mu:_mu;
    double s=(paramnew)?_new_sigma:_sigma;


    if(save_en)
    {
        cen.open(en_filename, ios::app);
        if(!cen.is_open())
        {
            cerr << "ERROR: Unable to open " << en_filename <<endl;
            clog << "ERROR: Unable to open " << en_filename <<endl;
        }
        cen << "# mu = "<< m<< ", sigma = "<<s<<endl;
        cen << "BLOCK, ACTUAL_E, E_AVE, ERR" << endl;
    }


    for (int i=0; i<_nblocks; i++) 
    {
        _block_avg=0;
        for(int j=0; j<_nsteps; j++)
        {
            move(m, s);
            if(writepos) write_position();
            _block_avg+=local_en(_x, m, s);
        }

        _block_avg /= _nsteps;
        avg += _block_avg;
        avg2+=_block_avg*_block_avg;
        _error = error(avg/(i+1), avg2/(i+1), i);

        if(save_en)
            cen << i+1<<", " 
                << _block_avg<<", " << avg/(i+1)<<", "
                << _error << endl;
    
        //progress(i, _nblocks, 30, "");
    }


    if(save_en) cen.close(); 

    clog << "FINAL ENERGY FOR CONFIGURATION ("<< m << ", " << s 
         << ") = " << avg/_nblocks << endl;    
    return avg/_nblocks;
}

void System::anneal_move()
{
    // generate new annealing parameter with initial distribution
    _new_mu=(_rnd.*annealing_distribution)(_mu, distr_width);
    _new_sigma=(_rnd.*annealing_distribution)(_sigma, distr_width);

    //compute new proposal energy
    _new_energy = compute_energy(true);
    

    if(this->anneal_metro())
    {
        this->accept_anneal_move();
        clog << "\tACCEPTED\n\n";
    }
    else{
        clog << "\tREJECTED\n\n";
    }

    _nanneal_attempts+=1;

    copt << _temp<<", " << _nanneal_attempts%(_anneal_nsteps+1)<<", " 
         << _mu<<", " << _sigma<<", " << _nblocks<<", " 
         << _energy<<", " << _error<< endl;
    
    return;
    
}


double System::compute_energy(string enfile)
{

    this->equilibration();

    avg=0;
    avg2=0;
    //_error=0;

    _naccepted=0;
    _naccepted=0;

    cen.open(enfile, ios::trunc);
    if(!cen.is_open())
    {
        cerr << "ERROR: Unable to open " << en_filename <<endl;
        clog << "ERROR: Unable to open " << en_filename <<endl;
    }
    cen << "# mu = "<< _mu<< ", sigma = "<<_sigma<<endl;
    cen << "BLOCK, ACTUAL_E, E_AVE, ERR" << endl;


    for (int i=0; i<_nblocks; i++) 
    {
        _block_avg=0;
        for(int j=0; j<_nsteps; j++)
        {
            move(_mu, _sigma);
            _block_avg+=local_en(_x, _mu, _sigma);
        }

        _block_avg /= _nsteps;
        avg += _block_avg;
        avg2+=_block_avg*_block_avg;
        _error = error(avg/(i+1), avg2/(i+1), i);

        cen << i+1<<", " 
            << _block_avg<<", " << avg/(i+1)<<", "
            << _error << endl;

        //progress(i, _nblocks, 30, "");
    }


    cen.close(); 

    clog << "FINAL ENERGY FOR CONFIGURATION ("<< _mu << ", " << _sigma 
         << ") = " << avg/_nblocks << endl;    
    return avg/_nblocks;
}


bool System::anneal_metro()
{
    bool decision=false;

    if(this->_new_energy<this->_energy) decision=true;
    else if (this->_rnd.Rannyu() < exp(-2*_beta*(_new_energy-_energy)) )
     decision=true;

    return decision;
}

void System::accept_anneal_move()
{
    _energy=_new_energy;
    _mu=_new_mu;
    _sigma=_new_sigma;
    _nanneal_accepted+=1;
    return;
}

void System::step_cooling()
{
    double newt=_temp-_temp_decr;
    if(newt<_min_temp)
        newt=_min_temp;
    if (_temp <= _min_temp) 
        _end=true;

    this->set_temperature(newt);
}

void System::step_cooling_beta()
{
    double newb=_beta+_temp_decr;
    if(newb>_max_beta)
        newb=_max_beta;
    if (_beta >= _max_beta) 
        _end=true;

    this->set_beta(newb);
}

void System::rate_cooling()
{
    double newt=_temp_decr*_temp;
    if(newt<_min_temp)
        newt = _min_temp;
    if (_temp <= _min_temp) 
        _end=true;


    this->set_temperature(newt);
}

void System::rate_cooling_beta()
{
    double newb=_temp_decr*_beta;
    if(newb>_max_beta)
        newb = _max_beta;
    if (_beta >= _max_beta) 
        _end=true;


    this->set_beta(newb);
}

void System::cooling()
{
    (this->*temp_decrease)();
}

void System::sample_probability(string filename, int npoints)
{
    ofstream sout(filename);
    if(!sout.is_open())
    {
        cerr << "ERROR: Unable to open the file "<< filename << endl;
        clog << "ERROR: Unable to open the file "<< filename << endl;
        exit(-1);
    }

    sout << "# mu = " << _mu << ", sigma = "<<_sigma<<endl;
    sout << "pos\n";
    for(int k=0; k<npoints; k++)
    {
        this->move(_mu, _sigma);
        sout << _x << endl;
    }

    sout.close();
    clog << "Sampled probability for WF at mu = "<<_mu << ", sigma = "<<_sigma<<endl; 
    return;

}

bool System::terminated()
{
    return _end;
}

void System::compute_initial_condition()
{
    string init_en_file = "OUTPUT/initial_energy.dat";
    _energy=this->compute_energy(init_en_file);
    _nanneal_accepted=0;
    _nanneal_attempts=0;
}

//set

void System::set_mu(double mu)
{
    this->_mu=mu;
    return;
}

void System::set_sigma(double sigma)
{
    this->_sigma=sigma;
    return;
}

void System::set_pos0(double x)
{
    this->_x0=x;
    return;
}

void System::set_energy_filename(string filename)
{
    this->en_filename=filename;
    return;
}
void System::set_optimization_filename(string filename)
{
    this->optimization_filename=filename;
    return;
}

void System::set_temperature(double temp)
{
    this->_temp=temp;
    this->_beta=1./temp;
}

void System::set_beta(double beta)
{
    this->_beta=beta;
    this->_temp=1./beta;
}


void System::set_min_temperature(double mintemp)
{
    this->_min_temp=mintemp;
    this->_max_beta=1./_min_temp;
}


//get

int System::get_n_annealing_steps()
{
    return this->_anneal_nsteps;
}

double System::get_temperature()
{
    return this->_temp;
}

void System::initialize(string inputfile)
{
    initialize_files();
    initialize_rnd();



    ifstream input(inputfile);

    if(!input.is_open())
    {
        cerr << "ERROR: Unable to open  input file: "<< inputfile << endl;
        clog << "ERROR: Unable to open  input file: "<< inputfile << endl;
    }

    // read input file
    string property;
    double delta;
    while ( !input.eof() ){
        input >> property;
        
        // SET SIMULATION TYPE  
        if( property == "INITIAL_POS" ){
            input >> _x0;
            clog << "PARTICLE POSITION INITIALIZED" << endl;

        }
        else if (property == "MU")
        {
            input >> _mu;
            clog << "MU INITIALIZED" << endl;
        }
        else if (property == "SIGMA") 
        {
            input >> _sigma;
            clog << "SIGMA INITIALIZED" << endl;
        }
        else if (property == "ANNEALSTEP_DIST" )
        {
            input >> distribution_type;
            if (distribution_type=="gauss" or distribution_type == "GAUSS")
                annealing_distribution = &Random::Gauss;
            else if (distribution_type=="uniform" or distribution_type == "UNIFORM")
                annealing_distribution = &Random::centered_Rannyu;
            input >> distr_width;

            clog << "ANNEALING METROPOLIS DISTRIBUTION = " << distribution_type 
                 << "  " << distr_width <<endl;

        }
        //INIT PROPERTIES
        else if( property == "TEMP" ){
            input >> _init_temp;
            _temp = _init_temp;
            _beta = 1.0/_temp;
            clog << "INITIAL TEMPERATURE = " << _temp << endl;

        }
        else if( property == "DELTA" ){
            input >> delta;
            clog << "DELTA = " << delta << endl;
            _delta = delta;
        } 
        else if( property == "NBLOCKS" ){
            input >> _nblocks;
            clog << "NBLOCKS= " << _nblocks << endl;

        } 
        else if( property == "NSTEPS" ){
            input >> _nsteps;
            clog << "NSTEPS= " << _nsteps << endl;
        } 
        else if( property == "ANNEAL_NSTEPS"){
        input >> _anneal_nsteps;
        clog << "ANNEALING NSTEPS PER TEMP = "<< _anneal_nsteps;
        } 
        else if( property == "EQSTEPS" ){
            input >> eq_steps;
            clog << eq_steps << " EQUILIBRATION STEPS" << endl;
        } 
        else if ( property == "COOLING_SCHEDULE")
        {
            input >> annealing_schedule;
            input >> _temp_decr;

            if(annealing_schedule == "step" || annealing_schedule == "STEP")
                temp_decrease = &System::step_cooling;

            else if(annealing_schedule == "rate" || annealing_schedule == "RATE")
            { 
                temp_decrease = &System::rate_cooling;
                if(_temp_decr>1)
                {
                    cerr<<"Input Error: the cooling rate for temperature must be minor than 1\n";
                    exit(1);
                }
            }
            else if(annealing_schedule == "step_beta" || annealing_schedule == "STEP_BETA")
                temp_decrease = &System::step_cooling_beta;

            else if(annealing_schedule == "rate_beta" || annealing_schedule == "RATE_BETA") 
            {
                temp_decrease = &System::rate_cooling_beta;

                if(_temp_decr<1)
                {
                    cerr<<"Input Error: the cooling rate for beta must be major than 1\n";
                    exit(1);
                }

            }

    
            clog << "ANNEALING SCHEDULE = "<< annealing_schedule 
                << " with parameter "<<_temp_decr <<endl;
        }
        else if(property == "MIN_TEMP")
        {
            input >>_min_temp;
            _max_beta = 1./_min_temp;

            clog << "MIN TEMP = "<< _min_temp
                 << ", MAX_BETA = "<< _max_beta<<"\n";
        }
        else if(property == "MAX_BETA")
        {
            input >>_max_beta;
            _min_temp = 1./_max_beta;

            clog << "MIN TEMP = "<< _min_temp
                 << ", MAX_BETA = "<< _max_beta<<"\n";
        }
        else if(property == "EN_FILENAME")
        {
            input >> en_filename;
            if (en_filename.find("OUTPUT/") ==  string::npos)
            {
                en_filename="OUTPUT/"+en_filename;
            }

        }
        else if(property == "OPT_FILENAME")
        {
            input >> optimization_filename;            
            if (optimization_filename.find("OUTPUT/") ==  string::npos)
            {
                optimization_filename="OUTPUT/"+optimization_filename;
            }
        }
        else if(property == "LOG_FILENAME")
        {
            input >> log_filename;
            if (log_filename.find("OUTPUT/") ==  string::npos)
            {
                log_filename="OUTPUT/"+log_filename;
            }
        }
        else if( property == "ENDINPUT" ){
            clog << "Reading input completed!\n\n" << endl;
            break;
        } 
        else 
        {
            cerr << "PROBLEM: unknown input" << endl;
            clog << "UNKNOWN INPUT: "<< property << endl;
        }
    }

    input.close();

    compute_initial_condition();
}


void System::write_position()
{
    ofstream cpos(wf_filename, ios::app);

    if(!cpos.is_open())
    {
        cerr << "ERROR: Unable to open " << wf_filename << endl;
    }

    cpos << _x << endl;

    return;
}