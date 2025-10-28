#include "operators.h"
#include "function.h"
#include "map.h"
#include "population.h"
#include "random.h"
#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

selection::selection()
{
    init_gen();
}

selection::selection(coordinates &map)
{
    this->_map = &map ;
    init_gen();

}

/*
Take in input the number of cities 
*/
selection::selection(int n_points, string path)
{
    this->_map = new coordinates(n_points, path);
    this->existing_coord=false;
    init_gen();

}

selection::selection(string filename)
{
    this->_map = new coordinates(filename);
    this->existing_coord=false;
    init_gen();
}

selection::~selection()
{
    //destroy pointer's content only if dinamically allocated
    if(!this->existing_coord)
        delete []_map;
}

double selection::evaluate_sol(crom sol)
{
    return (_map->*C)(sol);
}

void selection::compute_costs(vector<crom> &pop)
{
    _pop_costs.resize(pop.size());
    for(int k = 0; k<pop.size(); k++)
    {
        _pop_costs[k]=this->evaluate_sol(pop[k]);
    }
}


/*
merge sort based on the coordinates cost
*/
vector<crom> selection::sort_population(vector<crom> &pop)
{


    if (pop.size()>1) {
        int mid = pop.size()/2;
        
        vector<crom> lefthalf(pop.begin(),pop.begin()+mid);

        vector<crom> righthalf(pop.begin()+mid,pop.begin()+pop.size());

        lefthalf = sort_population(lefthalf);
        righthalf = sort_population(righthalf);

        unsigned i = 0;
        unsigned j = 0;
        unsigned k = 0;

        while (i < lefthalf.size() && j < righthalf.size()) {
            if (this->evaluate_sol(lefthalf[i]) < this->evaluate_sol(righthalf[j])) {
                pop[k]=lefthalf[i];
                i++;
            } else {
                pop[k] = righthalf[j];
                j++;
            }
            k++;
        }

        while (i<lefthalf.size()) {
            pop[k] = lefthalf[i];
            i++;
            k++;
        }

        while (j<righthalf.size()) {
            pop[k]=righthalf[j];
            j++;
            k++;
        }

    }

    return pop;
}


/*
Compute probabilities of the solutions and store it 
in the _pop_costs vector using the distance between cities
*/
void selection::compute_prob1(vector<crom> &pop)
{
    //store costs in _pop_costs vector
    this->compute_costs(pop);
    
    //
    double norm = sqrt(std::inner_product(
        this->_pop_costs.begin(), this->_pop_costs.end(),
        this->_pop_costs.begin(), 0.  
    ));

    

    // vector per scalar std multiplication to normalize _pop_costs 
    std::transform(
        this->_pop_costs.begin(), 
        this->_pop_costs.end(), 
        this->_pop_costs.begin(), //update _pop_costs
        std::bind(
            std::multiplies<double>(),
            std::placeholders::_1,
            1./norm
        )
    );
    
    //check that they are now a probability
    double newnorm = std::inner_product(
        this->_pop_costs.begin(), this->_pop_costs.end(),
        this->_pop_costs.begin(), 0.  
    );

    double eps=1e-5;
    if ((newnorm<1-eps)||(newnorm>1+eps))
    {
        cerr<<"Error: probability not normalized: quadratic sum ="
            << newnorm << endl;

        exit(1);
    }
    return;    
}

void selection::selection_prob1(vector<crom> &pop)
{
    compute_prob1(pop); // store in _pop_costs the probabilities

    vector<crom> newpop(pop.size());

    for(int j=0; j<pop.size(); j++)
    {
        double part_sum=0.;
        double r = this->_rnd.Rannyu();
    
        //check if the random value uniformly generated
        //  is inside the intervals
        for(int k=0; k<this->_pop_costs.size(); k++)
        {
            part_sum+=this->_pop_costs[k];
            if (r<part_sum)
            {
                newpop[j]=pop[k];
                break;
            }
        }
    }

    //overwrite population
    pop=newpop;

    return;
}

void selection::selection_prob2(vector<crom> &pop, double p)
{
    vector<crom> newpop (pop.size());

    for(int k=0; k<pop.size(); k++)
    {
        int j = static_cast<int>(
            pop.size()*pow(this->_rnd.Rannyu(), p)
        );
        newpop[k]= pop[j];
    }

    //overwrite pop
    pop = newpop;
    return;
}




void selection::init_gen(string primesfile, int primerow, string seedfile)
{

    // INIT RANDOM GENERATOR
    int p1, p2; // Read from ../INPUT/Primes a pair of numbers to be used to initialize the RNG
    ifstream Primes("Primes");
    for(int r=0; r<primerow-1; r++)
        Primes.ignore (
            std::numeric_limits<std::streamsize>::max(),
            '\n' 
        ); //skip primerow lines 
    Primes >> p1 >> p2 ;
    Primes.close();
    int seed[4]; // Read the seed of the RNG
    ifstream Seed("seed.in");
    Seed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
    this->_rnd.SetRandom(seed,p1,p2);
    return;
}


void selection::save_population_best(vector<crom> &pop, string filename)
{

    int best_index=0;
    double min_cost=this->evaluate_sol(pop[0]);
    for(int k=1; k<pop.size(); k++)
    {
        double cost=this->evaluate_sol(pop[k]);
        if(cost<min_cost)
        {
            min_cost=cost;
            best_index=k;
        }
    }

    ofstream out(filename, ios::app);
    for(int j:pop[best_index])
    {
        out <<j<<", ";
    }
    out<<endl;

    return;
}


/*####################################################*/
/*                        MUTATION                           */
mutation::mutation()
{
    init_gen();
};

mutation::~mutation()
{};


crom mutation::permutation(crom sol)
{
    int index1 = static_cast<int>(
        this->_rnd.Rannyu(0, sol.size())
    );

//    int index2 = static_cast<int>(
  //      this->_rnd.Rannyu(0, sol.size())
    //);

    //if(index1==index2) 
    //{
    int index2 ;
    if(this->_rnd.Rannyu()<0.5)
    {
        index2 = static_cast<int>(
            this->_rnd.Rannyu(0, index1)
        );    
    }
    else {
        index2 = static_cast<int>(
            this->_rnd.Rannyu(index1+1, sol.size())
        );
    }
    //}
    int temp=sol[index1];
    sol[index1]=sol[index2];
    sol[index2]=temp;

    return sol;
}


//redundant because yet modifies the pop 
//but in case i want to change let's keep it like that
vector<crom> mutation::permutation (
    vector<crom> &pop, double prob
)
{
    for(int k=0; k<pop.size(); k++)
    {
        if(this->_rnd.Rannyu()<prob)
        {
            pop[k]=this->permutation(pop[k]);
        }
    }

    return pop;
}


/*cyclic shifting of m sequntial number of m positions*/
crom mutation::shift(crom sol, int n, int m) 
{
    if(m>=sol.size())
    {
        cerr<<"Error: can't shift of a lenght major than vector lenght"<<endl;
        exit(1);
    }

    int N=sol.size()-1;

    n%=N;
    crom newsol=sol;
    int start = max(0, (m+n)-N);
    for(int j =1; j<=n; j++)
    {
        newsol[start+j]=sol[j+m];
    }

    // shift 
    for(int k=0; k<m; k++)
    {   
        newsol[1+(k+n)%N]=sol[k+1];
    }

    return newsol;
}


crom mutation::subpermutation(crom sol, int m)
{

    int N = sol.size();
    if(m>N/2)
    {
        cerr<<"Error: can't perform a permutation "
            <<"of group with this size"<<endl;

        exit(1);
    }

    crom newsol; 

//    int start = static_cast<int>(this->_rnd.Rannyu(
//        1, N-m
//    ));

    int start;

    if(m==N/2)
    {
        if ((N-1)%2==0)
        {
            newsol = swap(
                sol, 1, 
                (N-1)/2 +1,
                 (N-1)/2
            );
        }
        else { // 3 possibilities
            double r = this->_rnd.Rannyu();
            if (r<=0.5)
            {
                start = 1;
                if(r<0.25)
                {
                    newsol = swap(
                        sol, 
                        start,
                        (N)/2,
                        (N-1)/2   
                    );
                }else {
                    newsol = swap(
                        sol, 
                        start,
                        (N+1)/2,
                        (N-1)/2   
                    );
                }
            }
            else {
                start = 2;
                newsol = swap(
                    sol, 
                    start,
                    (N+1)/2,
                    (N-1)/2   
                );
            }            
        }
    }          


    return newsol;
}

crom mutation::swap(
    crom sol, int start1, int start2, int size
)
{
    if(start1>start2)
    {
        int temp = start2;
        start2=start1;
        start1=temp;
    }

    if(start1 + size > start2)
    {
        cerr << "Error: cannot swap it, there is a common subset" <<endl;
        exit(1);
    } 
    crom newsol=sol;
    for(int k=0; k<size; k++)
    {
        newsol[start1+k]=sol[start2+k];
        newsol[start2+k]=sol[start1+k];
    }
    return newsol;
}

void mutation::init_gen(string primesfile, int primerow, string seedfile)
{

    // INIT RANDOM GENERATOR
    int p1, p2; // Read from ../INPUT/Primes a pair of numbers to be used to initialize the RNG
    ifstream Primes("Primes");
    for(int r=0; r<primerow-1; r++)
        Primes.ignore (
            std::numeric_limits<std::streamsize>::max(),
            '\n' 
        ); //skip primerow lines 
    Primes >> p1 >> p2 ;
    Primes.close();
    int seed[4]; // Read the seed of the RNG
    ifstream Seed("seed.in");
    Seed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
    this->_rnd.SetRandom(seed,p1,p2);
    return;
}
