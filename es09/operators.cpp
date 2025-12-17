#include "operators.h"
#include "function.h"
#include "map.h"
#include "population.h"
#include "random.h"
#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

/*####################################################*/
/*                     SELECTION                      */
/*####################################################*/

/*
    Default constructor only initialize random generator 
 */
selection::selection(){}

/*
Initialize random generator and coordinates
*/
selection::selection(coordinates &map)
{
    this->_map = &map ;
}

/*
Initialize random generator and 
generate the cities along a chosen path     
*/
selection::selection(int n_points, string path)
{
    this->_map = new coordinates(n_points, path);
    this->existing_coord=false;

}

/*
Load cities from file and initialize random number generator
*/
selection::selection(string filename)
{
    this->_map = new coordinates(filename);
    this->existing_coord=false;
}

/*
If dinamically allocated delete pointer to coordinates
*/
selection::~selection()
{
    //destroy pointer's content only if dinamically allocated
    if(!this->existing_coord)
        delete []_map;
}

/*
    Select a Metric for the evaluation
    Parameter :
        - string norm 
            - "L1" absolute distance  
            - "L2" squared absolute distance
*/
void selection::set_metric(string norm)
{
    if (norm == "L1" || norm == "l1")
    {
        this->C = &coordinates::L1;
    }
    else if (norm == "L2" || norm == "l2")
    {
        this->C = &coordinates::L2;
    }
    else {
        cout << "Default metric L2" << endl; 
        this->C = &coordinates::L2;
    }
}


/*
Evaluate solution from a given metric
*/
double selection::evaluate_sol(crom sol)
{
    return (_map->*C)(sol);
}




/*####################################################*/
/*                        MUTATION                    */
/*####################################################*/


mutation::mutation(){
    init_gen(
        _rnd, "Primes", 
        2, "seed.in"
    );
}

mutation::~mutation()
{};

/*
Permute 2 random index in the given solution
*/
crom mutation::permutation(crom sol)
{
    int index1 = static_cast<int>(
        this->_rnd.Rannyu(1, sol.size())
    );

    int index2 ;

    if(this->_rnd.Rannyu()<static_cast<double>(index1)/(sol.size()-1))
    {
        index2 = static_cast<int>(
            this->_rnd.Rannyu(1, index1)
        );    
    }
    else {
        index2 = static_cast<int>(
            this->_rnd.Rannyu(index1+1, sol.size())
        );
    }
//    print(sol);
//    cout << index1<<"/"<<sol.size()<<"\n";
    int temp=sol[index1];
    sol[index1]=sol[index2];

    sol[index2]=temp;


    return sol;
}




/*cyclic shifting of m sequntial number of m positions*/
crom mutation::shift(crom sol, int n, int m) 
{
    if(m>=sol.size())
    {
        cerr<<"Error: can't shift of a lenght major than vector lenght"<<endl;
        exit(1);
    }

    int N=sol.size();
    
    n%=N;

    int index = static_cast<int>(this->_rnd.Rannyu(1, N-m));

    // if index + m + n outbound ciclic shift

    crom newsol=sol;

    if(index+n<N)
    {
        int upper_shift, bottom_shift;
        if (index + m +n - 1 >= N)
        {
            upper_shift   =  N -1;
            bottom_shift =  index + m + n - N; 
        }
        else {
            upper_shift = index + m + n-1;
            bottom_shift = 0;
        }
        for(int k =1; k<index; k++)
        {
            newsol[bottom_shift+k] = sol[k];
        }
        int newN = min(N-index-m, n); //number of el to move back
        for(int k =0; k<newN; k++)
        {
            newsol[bottom_shift+index + k] = sol[index+m+k];
        }
        //shifted
        for(int k=0; k<m; k++)
        {
            int newindex = (index+k+n>N-1)? (index+k+n-N+1):(index+k+n);
            newsol[newindex]=sol[index+k];
        }

    }
    else {
        int shift=N-n-1;
        for(int k=index; k<index+m; k++)
        {
            newsol[k-shift]=sol[k];
        }        
        for(int k=1; k<=shift; k++)
        {
            newsol[index-k+m]=sol[index-k];
        }    
    }
    return newsol;
}



// to finish

crom mutation::subpermutation(crom sol, int m)
{

    int N = sol.size();
    if(m>(N-1)/2)
    {
        cerr<<"Error: can't perform a permutation "
            <<"of group with this size"<<endl;

        exit(1);
    }

    crom newsol=sol; 

    int s1,s2;

    //generate the first index
    s1=static_cast<int>(_rnd.Rannyu(1,N-2*m+1)); 

    //generate s2
    s2=static_cast<int>(_rnd.Rannyu(s1+m, N-m+1));

    // cout <<"[";
    // for(int i=0;i<sol.size();i++)
    // {
    //     if(i==s1||i==s1+m || i==s2 || i==s2+m)
    //         cout <<"|";
    //     cout <<sol[i]<<" ";
    // }
    // cout <<"]\n";

    for(int k=0; k<m; k++)
    {
        newsol[s1+k]=sol[s2+k];
        newsol[s2+k]=sol[s1+k];
    }

    // cout <<"[";
    // for(int i=0;i<newsol.size();i++)
    // {
    //     if(i==s1||i==s1+m || i==s2 || i==s2+m)
    //         cout <<"|";
    //     cout <<newsol[i]<<" ";
    // }
    // cout <<"]\n";

    return newsol;
}


crom mutation::inversion(crom sol, int m)
{
    int start=static_cast<int>
        (_rnd.Rannyu(1, sol.size()-m));

    crom newsol = sol;
    for(int k=0; k<=m; k++)
    {
        newsol[start+k]=sol[start+m-k];
    }

    return newsol;
}




crom mutation::swap(
    crom sol, int start1, int start2, int size
)
{
    if(start1>start2)
    {
        int temp = start2;
        start2=start1;
        start1=temp;
    }

    if(start1 + size > start2)
    {
        cerr << "Error: cannot swap it, there is a common subset" <<endl;
        exit(1);
    } 
    crom newsol=sol;
    for(int k=0; k<size; k++)
    {
        newsol[start1+k]=sol[start2+k];
        newsol[start2+k]=sol[start1+k];
    }
    return newsol;
}



/******************************************************/
/*                   crossing                       */
/******************************************************/
crossing::crossing()
{
    init_gen(
        _rnd, "Primes", 
        3, "seed.in"
    );

}


pair<crom, crom> crossing::cross(crom parent1, crom parent2)
{
    int N = parent1.size();
    int crosspoint = static_cast<int> 
        (this->_rnd.Rannyu(1, N));
    


    crom support = parent2;
    
    for(int k=0; k< crosspoint; k++)
        // remove from parent2 copy the elements
        //that are yet in first part of parent 2
         support.erase(std::remove(
            support.begin(), 
            support.end(), 
            parent1[k]
        ));
    


    for(int k=0; k<support.size(); k++)
    {
        parent1[k+crosspoint] = support[k]; 
    }


    support = parent1;

    for(int k=0; k< crosspoint; k++)
        // remove from parent2 copy the elements
        //that are yet in first part of parent 2
         support.erase(std::remove(
            support.begin(), 
            support.end(), 
            parent2[k]
        ));

    // cout << "\tspurged parent1\n\t\t";
    // print(support);
    
    for(int k=0; k<support.size(); k++)
    {
        parent2[k+crosspoint] = support[k]; 
    }

    return {parent1, parent2};

}

/*
Crossinig operation between two sols
*/
pair<crom, crom> crossing::cross(
    crom parent1, crom parent2, int crosspoint
)
{
    int N = parent1.size();


    // cout << "crossing point "<< crosspoint<<"\n";


    cout << "parent1 = [";
    for(int k=0; k<parent1.size();k++)
    {
        if(k==crosspoint) cout << "| ";
        cout << parent1[k]<< " ";
    }
    cout << "]\n";

    cout << "parent2 = [";
    for(int k=0; k<parent2.size();k++)
    {
        if(k==crosspoint) cout << "| ";
        cout << parent2[k]<< " ";
    }
    cout << "]\n";

    crom support = parent2;
    
    for(int k=0; k< crosspoint; k++)
        // remove from parent2 copy the elements
        //that are yet in first part of parent 2
         support.erase(std::remove(
            support.begin(), 
            support.end(), 
            parent1[k]
        ));
    

    cout << "\tspurged parent2\n\t\t";
    print(support);

    for(int k=0; k<support.size(); k++)
    {
        parent1[k+crosspoint] = support[k]; 
    }


    support = parent1;

    for(int k=0; k< crosspoint; k++)
        // remove from parent2 copy the elements
        //that are yet in first part of parent 2
         support.erase(std::remove(
            support.begin(), 
            support.end(), 
            parent2[k]
        ));

    cout << "\tspurged parent1\n\t\t";
    print(support);
    
    for(int k=0; k<support.size(); k++)
    {
        parent2[k+crosspoint] = support[k]; 
    }

    return {parent1, parent2};

}


