#include "function.h"
#include "map.h"
#include "operators.h"
#include "population.h"
#include <iomanip>
#include <vector>

using namespace std;


int main()
{
/*    population pop(4, 10);
    cout << "Initial population " <<endl;
    pop.print();
*/

    int ncities=5, dimpop=10;

    coordinates coord(ncities, "circle");

    coord.print();

    population pop(coord, dimpop);



    //pop.print();
    selection sel(coord);
    vector<crom> sols=pop.get_pop();
    
    cout << "before sort"<<endl;
    for(crom s:sols)
    {
        for(int k:s)
            cout << k << " ";
        cout<<setw(18)<<"cost = "<<sel.evaluate_sol(s);
        cout<<endl;
    }

    sel.sort_population(sols);

    cout << "after sort"<<endl;
    for(crom s:sols)
    {
        for(int k:s)
            cout << k << " ";
        cout<<setw(18)<<"cost = "<<sel.evaluate_sol(s);
        cout<<endl;
    }

    vector<crom> sols2 = sols;


    cout << "New population for prob1" << endl;
    sel.selection_prob1(sols);

    for(crom s:sols)
    {
        for(int k:s)
            cout << k << " ";
        cout<<setw(18)<<"cost = "<<sel.evaluate_sol(s);
        cout<<endl;
    }


    mutation mut;


    cout<<"\n\nTest SHIFT"<<endl;
    crom test = {1,2,3,4,5};
    cout<<"Starting point\n";
    print(test);


    cout<<"shift first 2 of 2\n";
    crom test_shift = mut.shift(test, 2, 2);
    print(test_shift);

    cout<<"shift first 1 of two\n";
    test_shift = mut.shift(test, 2, 1);
    print(test_shift);

    cout<<"shift first 3 of 2\n";
    test_shift = mut.shift(test, 2, 3);
    print(test_shift);

    cout<<"shift first 2 of 3\n";
    test_shift = mut.shift(test, 3, 2);
    print(test_shift);

    cout<<"shift first 2 of 5\n";
    test_shift = mut.shift(test, 5, 2);
    print(test_shift);


    test = {1, 2, 3, 4, 5, 6, 7};
    cout<<"\n\nTest\n";
    print(test);

    for(int k =0; k<20; k++)
        mut.subpermutation(test, 3);
    return 0;

}