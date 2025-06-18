#ifndef __FUNC_H__
#define __FUNC_H__

#include <iostream>
#include <cmath>
#include "../Parallel_number_generator_20250306/random.h"
#include <vector>
#include <bits/stdc++.h>
#include <fstream>

using namespace std;

double error( const vector<double>& AV, const vector<double>& AV2, int n );

int* complement_seeds(
	string filename="../Parallel_number_generator_20250306/Primes",
	int row = 0
);

void set_seed(
	Random &rnd,
	string seedfile="../Parallel_number_generator_20250306/seed.in",
	string primesname="../Parallel_number_generator_20250306/Primes"
);

void set_numeric_seed(
	Random &rnd,
	int seed[4]=new int [4] {0,0,0,1}, //not beautiful for memory
	string primesname="../Parallel_number_generator_20250306/Primes"
);


void save_data(
	vector<int> &throws, 
	vector<double> &data, 
	vector<double> &err,  
	string filename=""
);

void save_data(
	vector<float> &data, 
	string filename=""
);

void save_data(
	vector<double> &data, 
	string filename=""
);



void progress(
	int block, 
	int nblock, 
	int barlenght
);


#endif
