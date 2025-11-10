#include "function.h"
#include "map.h"
#include "population.h"
#include <string>

using namespace std;


int main()
{
/*    population pop(4, 10);
    cout << "Initial population " <<endl;
    pop.print();
*/

    int ncities=34, dimpop=100;

    string path="square";
    // coordinates coord(ncities, path);
    string coord_file="coordinates.csv";
    coordinates coord(coord_file);

    coord.print();
    cout << "---end of coord --\n";

    population pop(coord, dimpop);
    //pop.print();


    ifstream read ("seed.in");

    string seed;
    getline(read, seed);

    // Using the erase, remove_if, and ::isspace functions.
    seed.erase(remove_if(seed.begin(),
     seed.end(), ::isspace),
        seed.end());
    cout << "seed = " << stoi(seed) <<"\n";

    pop.set_best_filename(
        "OUTPUT/best_sol_"
        +path+"_seed"
        + to_string(stoi(seed))+".dat"
    );
    //coord.save_coordinates("coordinates.csv");


    pop.set_avg_filename(
        "OUTPUT/avg_"
        +path+"_seed"
        + to_string(stoi(seed))+".dat"
    );


    pop.initialize_files();

    pop.set_mutation_probability(
        0.1, 
        0.1, 
        0.1,
        0.1
    );

    pop.set_crossing_probability(0.8);



    int nsteps=1000;
    cout << "Evolution started\n";
    for(int j=0; j<nsteps; j++)
    {
        //cout << "evolution "<<j<<endl;
        pop.evolve();
        pop.save_population_best();
        pop.save_avg_cost();
        progress(j, nsteps);
    }
    cout<<RESET<<endl;

    //pop.print();
    cout << "ended"<<endl;
    return 0;

}