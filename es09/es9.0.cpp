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

    int /*ncities=34,*/ dimpop=100;

    // string path="circle";
    // coordinates coord(ncities, path);
    // string coord_file="coordinates.csv";
    coordinates coord;//(coord_file);

    coord.initialize();

    coord.print();
    cout << "---end of coord --\n";

    population pop(coord);
    // population pop(coord, dimpop);
    //pop.print();


    ifstream read ("seed.in");

    string seed;
    getline(read, seed);

    // Using the erase, remove_if, and ::isspace functions.
    seed.erase(remove_if(seed.begin(),
     seed.end(), ::isspace),
        seed.end());
    cout << "seed = " << stoi(seed) <<"\n";

    pop.initialize();


    pop.perform_evolution();
    return 0;

}