#include "function.h" 



void print(crom s, string prefactor)
{
    for(int k:s)
        cout <<prefactor<< k << " ";
    cout<<endl;
    return;
}

void print(vector<crom> &pop, string prefactor)
{
    for(crom s:pop)
        print(s, prefactor);
    return;
}


