#include <iostream>
#include "../es1/func.h"

double integrand(double x)
{
	return M_PI*0.5*cos(M_PI*x*0.5);
}

int main(int argc, char* argv[])
{
	//set random generator
	Random rnd;
	set_seed(rnd);
		
	int M= 10000*100; //total number of throws
	
	int N=100; //number of blocks
	
	int L=M/N; //number of throws per block
	
	vector<double> avg(N, 0);
	vector<double> av2(N, 0);

	vector<double> sum_prog(N, 0);
	vector<double> su2_prog(N, 0);
	vector<double> err_prog(N, 0);

	//vector of throws
	vector<int> thr(N);
	iota(thr.begin(), thr.end(), 0);
	
	//iterate on blocks
	for(int k=0; k<N; ++k)
	{
		double sum1=0.;
		for(int j=0; j<L; ++j)
		{
			//generate uniform 
			double r=rnd.Rannyu();
			sum1+=integrand(r);
		}
		avg[k]=sum1/L;
		av2[k]=avg[k]*avg[k];
		
	}
	
	// now progressive mean on the blocks values 
	for(int k=0; k<N; ++k)
	{
		for (int j=0; j<=k; ++j)
		{
			sum_prog[k]+=avg[j];
			su2_prog[k]+=av2[j];
		} 
		sum_prog[k]/=(k+1);
		su2_prog[k]/=(k+1);
		err_prog[k]=error(sum_prog, su2_prog, k);
	}
	
	for(int &el: thr )
	{
		el*=L;
	}
	
	string datafile="data/cosinteg_M"+to_string(M)+".csv";
	save_data(thr, sum_prog, err_prog, datafile);
	
	
	return 0;
}
