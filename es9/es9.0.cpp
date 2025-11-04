#include "function.h"
#include "map.h"
#include "population.h"

using namespace std;


int main()
{
/*    population pop(4, 10);
    cout << "Initial population " <<endl;
    pop.print();
*/

    int ncities=10, dimpop=100;

    coordinates coord(ncities, "circle");

    coord.print();
    cout << "---end of coord --\n";
    coord.save_coordinates("coordinates.csv");

    population pop(coord, dimpop);
    //pop.print();

    pop.set_mutation_probability(
        0.8, 
        0.8, 
        0.8
    );



    int nsteps=10000;
    cout << "Evolution started\n";
    for(int j=0; j<nsteps; j++)
    {
        //cout << "evolution "<<j<<endl;
        pop.evolve();
        pop.save_population_best();
        progress(j, nsteps);
    }
    cout<<RESET<<endl;

    //pop.print();
    cout << "ended"<<endl;
    return 0;

}