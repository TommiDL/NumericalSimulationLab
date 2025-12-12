#include <iostream>
#include "func.h"

using namespace std;

int main(int argc, char* argv[])
{
	//subinterval
	int M=100;
	
	//throws	
	int n=10000;

	int Ndist=100;

	Random rnd;
	set_seed(rnd); //read seed from file
	
	//store results for every distribution 
	vector<float> chi_vec(Ndist, 0);
	
	//iterate on the distributions
	for(int k=0; k<Ndist; ++k)
	{
		
		float chi2=0.;
		//for every subinterval save number of point detected
		vector<int> ni(M, 0);
		
		//generate points
		for (int i=0; i<n; ++i)
		{
			float r=rnd.Rannyu();
			//this is problematic if i'm not sure r  is between 0 and 1
			for(int j=0; j<M; ++j)  
			{
				//check if inside interval ---> subtitute with binary search
				if (r<(j+1)/static_cast<float>(M))
				{
					ni[j]+=1;
					break;
				}
			}		
		}
		
		
		float nM=n/static_cast<float>(M);

		//compute chi2		
		for (int j=0; j<M; ++j)
		{
			chi2+=pow(ni[j]-nM, 2);
		}
		chi2/=(nM);
		
		chi_vec[k]=chi2;
	}
	
	save_data(chi_vec, "data/chi2.csv");
	return 0;
}
