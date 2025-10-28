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

        ~population() {;};

        void print();
        int get_size() {return _pop_size;};
        vector<crom> get_pop(){return _population;};
    private:

        int _pop_size=10;
        vector<crom> _population;
        selection sel;

        bool check_population();
        bool check_solution(int i);
        
        //eventually change into L1 or another coordinates cost function



        

};

#endif