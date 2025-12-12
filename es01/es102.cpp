#include <iostream>
#include <cmath>


#include "func.h"
using namespace std;


int main(int argc, char* argv[])
{
	if (argc< 2)
	{
		cerr<<"Error: Program usage: "<<endl
		    <<"\t" << argv[0] << " <M [number of data to generate]>"<<endl;
	   	return 1;
		
	}
	int M=1000000; // total number of throws
	int N=atoi(argv[1]); // number of blocks
	
	int L=M/N; //number of throws per block
	
	// random generator
	Random rnd;
	set_seed(rnd); //read seed from file

	//vector of throws	
	vector<int> x(N); //[0, ...N-1]
	iota(x.begin(), x.end(), 0);


	if(x[x.size()-1]!=N-1) //check if correct
	{
		cerr<<x[x.size()-1];
		cerr<<"Error in vector initialization"<<endl;
		return -1;
	}


		
	vector<double> ave (N, 0); //averages on blocks
	vector<double> av2 (N, 0); //averages squared on blocks
	
	vector<double> sum_prog (N, 0);
	vector<double> su2_prog (N, 0);
	vector<double> err_prog (N, 0);

	//cout<<"computing averages"<<endl;
	for(int i=0; i<N; ++i) //iterate on blocks
	{
		float sum1=0; 
		//float sum2=0;
		for(int j=0; j<L; ++j) //iterate inside block
		{
			//int k=j+i*L;
			//generate uniform distribution
			float r=rnd.Rannyu();
			sum1+=pow(r-0.5, 2);	      //
			//sum2+=r[k]*r[k]; //prova
		}
		ave[i] = sum1/L; //average on block
		//av2[i]=sum2/L; //prova
		av2[i]=pow(ave[i], 2); 
		
		
	}

	//cout<<"computing integral	"<<endl;	
	for(int i=0; i<N; ++i)
	{
		for(int j=0; j<=i; ++j)
		{
			sum_prog[i]+=ave[j]; // sum j=0,i r_j
			su2_prog[i]+=av2[j]; // sum j=0,i (r_j)^2
		}
		sum_prog[i]/=(i+1);
		su2_prog[i]/=(i+1);
		err_prog[i] = error(sum_prog, su2_prog, i);
	}
	
	
	for (auto & element : x)
		element*=L; //number of throw = block * (number of throw for every block)
	
	
	
	string datafile="data/var_estimation_M"+to_string(M)+".csv";
	
	save_data(x, sum_prog, err_prog, datafile);
	
	
	

	return 0;
}
