#ifndef __function__
#define __function__

#include <iomanip>
#include <iostream>
#include <vector>
#include "random.h"
#include <fstream>

#define RESET   "\033[0m"

using namespace std;


typedef vector<int> crom;


void print(crom s, string prefactor="");
void print(vector<crom> &pop, string prefactor="");
void progress(int step, int nstep, int barlenght=30);

void init_gen(
    Random &rnd,
    string primesfile="Primes",
    int primerow=1, 
    string seedfile="seed.in"
);

void init_gen_seed(
    Random &rnd, 
    int *seed,
    int primerow=1, 
    string primesfile="Primes"
);

#endif