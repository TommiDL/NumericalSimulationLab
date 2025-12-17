#include "probability.h"
using namespace std;

double norm(const vector<double> &x)
{
	double norm=0;
	for(long unsigned int k=0; k<x.size(); ++k)
	{
		norm+=x[k]*x[k];
	}
	return sqrt(norm);
}

double p_1s(const vector<double> &x)
{
	if (x.size()!=3)
	{
		cerr<<"ERROR::point not 3d"<<endl;
		return 0;
	}

	double r= norm(x); 

	return exp(-2*r);//(1/M_PI)*exp(-2*r); 
	//normalization not necessary for metropolis
}



/*
	Probability function of a 2p orbital with m=0
	Parameter:
		- vector of 3 double x for position 
*/
double p_2p(const vector<double> &x) 
{
	if (x.size()!=3)
	{
		cerr<<"ERROR::point not 3d"<<endl;
		return 0;
	}
	double r=norm(x);

	
	return r*r*exp(-r)*x[2]*x[2]/(r*r);//(1/32*M_PI)*r*r*exp(-r)*cos(th)*cos(th);
	//normalization not necessary for metropolis
}

/*
	Probability function of a 2p orbital with m=1
	Parameter:
		- vector of 3 double x for position 
*/
double p_2pm1(const vector<double> &x) 
{
	if (x.size()!=3)
	{
		cerr<<"ERROR::point not 3d"<<endl;
		return 0;
	}
	double r=norm(x);
	
	return r*r*exp(-r)*(1-x[2]*x[2]/(r*r));//(1/32*M_PI)*r*r*exp(-r)*cos(th)*cos(th);
	//normalization not necessary for metropolis
}


double p_2s(const vector<double> &x)
{
	double r=norm(x);
	return pow(2-r , 2)*exp(-r);
	
}

double p_3s(const vector<double> &x)
{
	double r=norm(x);
	return pow(27-18*r +2*r*r, 2)*exp(-r);
	
}

double p_3pm0(const vector<double> &x)
{
	double r=norm(x);
	return 	r*r*exp(-1.5*r)*pow((6-r), 2)*x[2]*x[2]/(r*r);
}


double p_3dm0(const vector<double> &x)
{
	double r=norm(x);
	return 	r*r*exp(-1.5*r)*pow(3*x[2]*x[2]/(r*r) -1, 2);
}




