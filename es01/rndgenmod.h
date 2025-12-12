#include "../Parallel_number_generator_20250306/random.h"
#include <cmath>
using namespace std;

class RandomMod: public Random
{
	public:
		double exp(double lambda=1.);
		
		double CauchyLorentz(double mu=0., double gamma=1.);
};
