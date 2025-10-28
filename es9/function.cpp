#include "function.h" 



void print(crom s)
{
    for(int k:s)
        cout << k << " ";
    cout<<endl;
    return;
}

void print(vector<crom> &pop)
{
    for(crom s:pop)
        print(s);
    return;
}


