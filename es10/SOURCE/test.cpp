#include "function.h"
#include "map.h"
#include "operators.h"
#include "population.h"
#include "random.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace std;


int main()
{

    int ncities=8, dimpop=10;

    string coord_file="coord_trial.txt";
    coordinates coord(coord_file);
    coord.print();

    crom sol ={1,2,3,4, 5};
 
    selection sel(coord);
    double cost = sel.evaluate_sol(sol);

    print(sol);
    cout << "L2 cost = " << cost <<"\n";

    sel.set_metric("L1");
    cost = sel.evaluate_sol(sol);
    cout << "L1 cost = " << cost <<"\n";


    return 0;

}