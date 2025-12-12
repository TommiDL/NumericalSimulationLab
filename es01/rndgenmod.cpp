#include "rndgenmod.h"

double RandomMod::exp(double lambda)
{
	double y=Rannyu();
	return -(1/lambda)*log(1-y);
}

double RandomMod::CauchyLorentz(double mu, double gamma)
{
	double y=Rannyu();
	return mu+gamma*(tan(M_PI*(y-0.5)));
}

