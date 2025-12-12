#include <iostream>
#include "func.h"
#include "rndgenmod.h"

using namespace std;

int main(int arc, char* argv [])
{
		// random generator
	RandomMod rnd;
	set_seed(rnd); //read seed from file
	
	int n=10000;
	
	vector<int> Nl={1, 2, 10, 100}; 
	
	
	
	
	for (int &N: Nl)
	{
		// int L=n/N;
		vector<double> SN_unf(n, 0);
		vector<double> SN_exp(n, 0);
		vector<double> SN_chl(n, 0);
	

		for (int k=0; k<n; ++k)
		{
			for(int i=0; i<N; ++i)
			{
				SN_unf[k]+=rnd.Rannyu();
				SN_exp[k]+=rnd.exp();
				SN_chl[k]+=rnd.CauchyLorentz();
			}
			SN_unf[k]/=N;
			SN_exp[k]/=N;
			SN_chl[k]/=N;
		}
		cout<<"\t--> N = "<<N<<" <--"<<endl;
		save_data(SN_unf, "data/unif_N"+ to_string(N) +".csv");
		save_data(SN_exp, "data/exp_N"+ to_string(N) +".csv");		
		save_data(SN_chl, "data/cauchy_N"+ to_string(N) +".csv");
		cout<<endl;	
	}

	
	return 0;
}
