#ifndef __RANDOM_WALK_H__
#define __RANDOM_WALK_H__

#include <iostream>
#include <vector>
#include "../es01/func.h"

using namespace std;


/*
Random walk on lattice. the steps are made only in one of the three directions x, y, z
*/
class RW_lattice
{
	public:
		//constructor
		RW_lattice(); //default constructor
		RW_lattice(int* seed); //set seed of random number generator
		RW_lattice(double a); //set step and default 
					      //seed for random number generator
		RW_lattice(double a, int* seed);
		RW_lattice(double a, string seedfile);
		
		//destructor
		~RW_lattice(){};
		
		//methods
		vector<double> get_pos () const{return pos;};
		int get_nstep() const {return step;};
		
		void reset_rw();
		
		//make a step in random direction and verse on the lattice
		void make_step();
		
	protected:
		double m_a;
		vector<double> pos=vector <double>(3, 0.0);
		
		Random rnd;
		int step;
		
		
		double verse_increment();
		vector<double> dir_increment(double p_x=1/3., double p_y=1/3., double p_z=1/3.);
};

/*
Random walk. 
The steps are made following a random direction obtained by sampling uniformly the solid angle
*/
class RW_continuous
{
	public:
		//constructor
		RW_continuous(); //default constructor
		RW_continuous(int* seed); //set seed of random number generator
		RW_continuous(double a); //set step and default 
					      //seed for random number generator
		RW_continuous(double a, int* seed);
		RW_continuous(double a, string seedfile);
		
		//destructor
		~RW_continuous(){};
		
		//methods
		vector<double> get_pos () const{return pos;};
		int get_nstep() const {return step;};
		
		void reset_rw();
		
		//make a step in random direction and verse on the lattice
		void make_step();
		
	protected:
		double m_a;
		vector<double> pos=vector <double>(3, 0.0);
		
		Random rnd;
		int step;
		
		
		double verse_increment();
		vector<double> dir_increment(double p_x=1/3., double p_y=1/3., double p_z=1/3.);
};


#endif
