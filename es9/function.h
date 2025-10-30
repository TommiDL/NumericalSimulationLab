#ifndef __function__
#define __function__

#include <iomanip>
#include <iostream>
#include <vector>


using namespace std;


typedef vector<int> crom;


void print(crom s, string prefactor="");
void print(vector<crom> &pop, string prefactor="");


#endif