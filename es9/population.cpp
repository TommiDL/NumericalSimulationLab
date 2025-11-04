#include "population.h"
#include "function.h"
#include "map.h"
#include "operators.h"
#include <algorithm>
#include <armadillo>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <set>
#include <vector>
 

population::population(coordinates &map, int n): sel(map)
{
    init_gen(_rnd);

    this->_pop_size=n;

    crom ordered (map.get_ncities());

    //fill 
    std::iota(ordered.begin(), ordered.end(), 1);

    _population = vector<crom> (_pop_size);
    _new_population = vector<crom> (_pop_size);
    _pop_costs = vector<double>(_pop_size);

    for(int i=0; i<n; i++)
    {
        random_shuffle(ordered.begin()+1, ordered.end());
        _population[i]=ordered;

    }

    if(!check_population())
    {
        cerr<< "Error: population generated not correctly."
            <<" Repetition are present inside some solutions" << endl;
        exit(1);
    }

    //initialize best file
    out.open(best_filename);
    out << "# best solutions" << endl;
    
}


population::~population() 
{
    if(out.is_open())
        out.close();
};



bool population::check_population(bool _new)
{

    bool pop_ok=true;
    for(int i=0; i<_pop_size; i++)
    {
        if (!this->check_solution(i, _new))
        {
            cerr << "Population corrupted at index "<<i<<"\n";

            print("", _new);
            return false;
        }    
    }

    return true;
}


bool population::check_solution(int i, bool _new)
{
    // create a copy without the repetitions 
    // and check if the vectors have the same size

    vector<crom> *pop = (_new)?&_new_population:&_population;
    if ((*pop)[i][0] != 1)
    {        
        cerr <<"First index changed\n";
        return false;
    }    
    set<int> s;
    unsigned size = (*pop)[i].size();
    for( unsigned k = 0; k < size; ++k ) s.insert( (*pop)[i][k] );

    bool wasUnique = (s.size()==(*pop)[i].size());
    return wasUnique;
}



void population::print(string prefactor, bool _new)
{
    if(_new)
    {
        for(crom i: this->_new_population)
        {
            cout <<prefactor<< "[ ";
            for(int k: i)
                cout << k<< " ";
            cout <<"]\t"<<sel.evaluate_sol(i)<<"\n";
        }
        return;
    }

    for(crom i: this->_population)
    {
        cout <<prefactor<< "[ ";
        for(int k: i)
            cout << k<< " ";
            cout <<"]\t"<<sel.evaluate_sol(i)<<"\n";
    }
    return;
}



/*
Compute cost of a vector of cromosomes
*/
void population::compute_costs()
{
     
    _pop_costs.resize(_population.size());
    for(int k = 0; k<_population.size(); k++)
    {
        _pop_costs[k]=this->sel.evaluate_sol(
            _population[k]
        );
    }
}

/*
Return cost of i-th member of the population
To execute only after compute_prob1  
*/
double population::evaluate_sol(int i)
{
    return this->_pop_costs[i];
}



/*
Compute probabilities of the solutions and store it 
in the _pop_costs vector using the distance between cities
*/
void population::compute_prob1()
{

    //store costs in _pop_costs vector
    this->compute_costs();
    
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


int population::selection_prob1() //O(N)
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
            return k;
        }
    }

    return _pop_size; // if it fail
}



int population::selection_prob2(double p)   //O(1)
{
    
    int k = static_cast<int>(
        _population.size()*pow(this->_rnd.Rannyu(), p)
    );
    return k;
}


void population::save_population_best()
{

    int best_index=0;
    double min_cost=this->sel.evaluate_sol(_population[0]);
    for(int k=1; k<_population.size(); k++)
    {
        double cost=this->sel.evaluate_sol(_population[k]);
        if(cost<min_cost)
        {
            min_cost=cost;
            best_index=k;
        }
    }

    ofstream out(best_filename, ios::app);
    for(int j:_population[best_index])
    {
        out <<j<<", ";
    }
    out<<_population[best_index][0]<<","<<min_cost;
    out<<endl;

    return;
}




void population::inversion(int k)
{
    int m; //generated random every time
    if(this->_rnd.Rannyu()<mutation_prob[0])
    {

        m = this->_rnd.Rannyu(
            1, _new_population[k].size()-1
        );

        _new_population[k] = this->mut.inversion(_new_population[k], m);
    }

    return;
    
}



void population::permutation (int k)
{
    if(this->_rnd.Rannyu()<mutation_prob[1])
    {
        crom newsol=this->mut.permutation(_new_population[k]);
        _new_population[k]=newsol;
    }

    return;
}

void population::shift(int k)
{
    int m, n; // generated random every time
    if(this->_rnd.Rannyu()<mutation_prob[2])
    {


        m = this->_rnd.Rannyu(
            1, _new_population[k].size()-1
        );
        
        n =  this->_rnd.Rannyu(
            1, _new_population[k].size()
        );


        crom newsol =this->mut.shift(_new_population[k], n=n, m=m); 
        _new_population[k]=newsol;
    }

    return ;

}


void population::evolve()
{

    this->sort_population();
    this->compute_prob1();

    for(int k=0; k<_population.size()/2; k++)
    {
        // cout << "evolution iteration "<<k<<endl;
        int index1=(this->*prob)(),
            index2=(this->*prob)();
    
        // cout << "index 1" << index1<<"\n"
        //      << "index 2" << index2<<"\n"
        //      <<"size "<<_population.size()<<"\n";

        //crossing
        if(this->_rnd.Rannyu()<cross_prob) //crossing 
        {
            //make crossing on different indexes
            while(index2==index1)
            {
                //possible infinite looop
                index2=(this->*prob)();
            }
            pair<crom,crom> sons = cross.cross(
                _population[index1], 
                _population[index2]
            );
            
            //subsitute with sons
            _new_population[2*k] = sons.first;
            _new_population[2*k+1]=sons.second;
            // cout << "crossing done\n";
            // this->check_solution(2*k, true);
            // this->check_solution(2*k+1, true);
            
        }
        else {
            _new_population[2*k]=_population[index1];
            _new_population[2*k+1]=_population[index1];
        }

        // mutate the new population 
        // members regardless of the fact that they are 
        // crossed or not

        //inversion
        this->inversion(2*k);
        this->inversion(2*k+1);

        // permutation
        this->permutation(2*k);
        this->permutation(2*k+1);

        // shift
        this->shift(2*k);
        this->shift(2*k+1);


    }
    if(this->check_population(true))
    {
        //accept new pop and reset all 
        this->_population = _new_population;
        this->_new_population=vector<crom> (_pop_size);
        this->_pop_costs = vector<double> (_pop_size);
    }else
    {
        
        exit(1);
    }

    

    return;
}


/*
merge sort based on the coordinates cost
*/
void population::sort_population()
{
    this->_population = merge_cost(this->_population);
    return;
}

/*
merge sort based on the coordinates cost
*/
vector<crom> population::merge_cost(vector<crom> &pop)
{

    
    if (pop.size()>1) {
        int mid = pop.size()/2;
        
        vector<crom> lefthalf(pop.begin(),pop.begin()+mid);

        vector<crom> righthalf(pop.begin()+mid,pop.begin()+pop.size());

        lefthalf = merge_cost(lefthalf);
        righthalf = merge_cost(righthalf);

        unsigned i = 0;
        unsigned j = 0;
        unsigned k = 0;

        while (i < lefthalf.size() && j < righthalf.size()) {
            if (this->sel.evaluate_sol(lefthalf[i]) < this->sel.evaluate_sol(righthalf[j])) {
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


void population::set_mutation_probability(
    double p_inversion,
    double p_permutation,
    double p_shift 
)
{
    mutation_prob[0]=p_inversion;
    mutation_prob[1]=p_permutation;
    mutation_prob[2]=p_shift;

}




