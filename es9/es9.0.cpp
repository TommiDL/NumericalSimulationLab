#include "map.h"
#include "population.h"

using namespace std;


int main()
{
/*    population pop(4, 10);
    cout << "Initial population " <<endl;
    pop.print();
*/

    int ncities=34, dimpop=100;

    coordinates coord(ncities, "circle");

    coord.print();
    coord.save_coordinates("coordinates.csv");

    population pop(coord, dimpop);
    pop.print();

    pop.set_mutation_probability(
        0.8, 
        0.8, 
        0.8
    );



    for(int j=0; j<1000; j++)
    {
        cout << "evolution "<<j<<endl;
        pop.evolve();
        pop.print("\t");
        pop.save_population_best();
    }
    cout<<endl;

    pop.print();
    cout << "ended"<<endl;
    return 0;

}