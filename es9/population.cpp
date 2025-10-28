#include "population.h"
#include "map.h"
#include "operators.h"
#include <algorithm>
#include <armadillo>
#include <ostream>
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

}


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
    // create a copy without the repetitions and check if the vectors have the same size
    vector<int>::iterator it = unique(_population[i].begin(), _population[i].end());
    bool wasUnique = (it == _population[i].end());
    return wasUnique;
}



void population::print()
{
    cout << "Population:" << endl;
    for(crom i: this->_population)
    {
        cout << "[ ";
        for(int k: i)
            cout << k<< " ";
        cout <<"]\n";
    }

}



