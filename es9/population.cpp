#include "population.h"
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
    this->_pop_size=n;

    crom ordered (map.get_ncities());

    //fill 
    std::iota(ordered.begin(), ordered.end(), 1);

    _population = vector<crom> (_pop_size);

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

bool population::check_population()
{
    bool pop_ok=true;
    for(int i=0; i< _population.size(); i++)
    {
        pop_ok = pop_ok and this->check_solution(i);
    }

    return pop_ok;
}




bool population::check_solution(int i)
{
    // create a copy without the repetitions 
    // and check if the vectors have the same size

    if (_population[i][0] != 1)
        return false;

    
    set<int> s;
    unsigned size = _population[i].size();
    for( unsigned k = 0; k < size; ++k ) s.insert( _population[i][k] );

    bool wasUnique = (s.size()==_population[i].size());
    return wasUnique;
}



void population::print(string prefactor)
{
    for(crom i: this->_population)
    {
        cout <<prefactor<< "[ ";
        for(int k: i)
            cout << k<< " ";
        cout <<"]\n";
    }

}

void population::evolve()
{
    double prob=0.08;


    mut.inversion(
        this->_population, mutation_prob[0]
    );

    mut.permutation(
        this->_population, mutation_prob[1]
    );

    mut.shift(
        this->_population, mutation_prob[2]
    );

    if(this->check_population())
        sel.selection_prob1(this->_population);
    else
    {   
        int corr;
        cerr<<"Error: population corrupted"<<endl;
        for(int k =0; k<_population.size(); k++)
            if(!check_solution(k))
            {
                corr=k;
                break;
            }

            cerr << "Corruption at index " << corr <<endl;
            exit(1);
    }
     
    return;
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


void population::save_population_best()
{
    this->sel.save_population_best(
        this->_population,
        this->best_filename
    );
    return;
}
