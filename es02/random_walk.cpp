#include "random_walk.h"


//RW_lattice::

RW_lattice::RW_lattice()
{
//	pos=vector<double> (3,0);
	m_a=1.;
	set_seed(rnd);
	step=0;
}

RW_lattice::RW_lattice(int* seed)
{
	m_a=1.;
	set_numeric_seed(rnd, seed);
	step=0;
}

RW_lattice::RW_lattice(double a)
{
	m_a=a;
	set_seed(rnd);
	step=0;
}

RW_lattice::RW_lattice(double a, int* seed)
{
	m_a=a;
	set_numeric_seed(rnd, seed);
	step=0;
}

RW_lattice::RW_lattice(double a, string seedfile)
{
	m_a=a;
	set_seed(rnd, seedfile);
	step=0;
}


void RW_lattice::make_step()
{
	//select direction x | y | z
	vector <double> stepv = dir_increment();		
		
	// select verse Rx | Lx
	double verse=verse_increment();	
	
	// generate step of lenght a and update position and step number
	for(int i=0; i<3; ++i)
	{
		pos[i]+=stepv[i]*verse;
	}
	
	step+=1;
}


void RW_lattice::reset_rw()
{
	pos=vector<double> (3, 0.);
	step=0;
}

double RW_lattice::verse_increment()
{
	bool dir=(rnd.Rannyu()-0.5)>0;
	if (dir)	
		return m_a;
	
	else
		return -m_a;
	
}

//return copy but 3dim not heavy
vector<double> RW_lattice::dir_increment(double p_x, double p_y, double p_z)
{
	double r = rnd.Rannyu();
	if(r<p_x)
		return vector<double> {1., 0., 0.};
	else if(r<p_x+p_y)
		return vector<double> {0., 1., 0.};
	else
		return vector<double> {0., 0., 1.};
		
	
}


/*---------------------------------------------------------------------------------------------*//*----------------------------------------------------------------------------------------------*/


RW_continuous::RW_continuous()
{
//	pos=vector<double> (3,0);
	m_a=1.;
	set_seed(rnd);
	step=0;
}

RW_continuous::RW_continuous(int* seed)
{
	m_a=1.;
	set_numeric_seed(rnd, seed);
	step=0;
}

RW_continuous::RW_continuous(double a)
{
	m_a=a;
	set_seed(rnd);
	step=0;
}

RW_continuous::RW_continuous(double a, int* seed)
{
	m_a=a;
	set_numeric_seed(rnd, seed);
	step=0;
}

RW_continuous::RW_continuous(double a, string seedfile)
{
	m_a=a;
	set_seed(rnd, seedfile);
	step=0;
}


void RW_continuous::make_step()
{
	//select direction 
	vector <double> stepv = dir_increment();		
	double verse=verse_increment();
	
	// generate step of lenght a and update position and step number
	for(int i=0; i<3; ++i)
	{
		pos[i]+=stepv[i]*verse;
	}
	
	step+=1;
}


void RW_continuous::reset_rw()
{
	pos=vector<double> (3, 0.);
	step=0;
}


//return copy but 3dim not heavy
vector<double> RW_continuous::dir_increment(double p_x, double p_y, double p_z)
{
	double phi=rnd.Rannyu();
	double r = rnd.Rannyu();
	double theta=acos(1-2*r);

	return vector<double> {
		sin(theta)*cos(phi),
		sin(theta)*sin(phi),
		cos(theta)
	};
	
	
}

double RW_continuous::verse_increment()
{
	bool dir=(rnd.Rannyu()-0.5)>0;
	if (dir)
		return m_a;
	
	else
		return -m_a;
	
}


