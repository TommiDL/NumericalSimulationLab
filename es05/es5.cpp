#include "../es01/func.h"
#include "probability.h"
#include <fstream>

using namespace std;

//define type probability function
typedef double(*prob)(const vector<double>&);


/*
Metropolis algorithm
Parameters:
Return:
*/
vector<double> metropolis(
	const vector <double> x0, 
	prob p, 
	Random& rnd, double sigma,
	vector<float> &acceptance_rate,
	string flag="gauss"
)
{
	vector<double> x (3); 

	// generate new point using flag distribution
	if ((flag=="gauss") || (flag=="Gauss") || (flag=="GAUSS"))
	{
		x[0] = rnd.Gauss(x0[0], sigma);
		x[1] = rnd.Gauss(x0[1], sigma);
		x[2] = rnd.Gauss(x0[2], sigma);
	}
	else if((flag=="uniform") || (flag=="Uniform") || (flag=="UNIFORM")) 
	{
		x[0] = rnd.Rannyu(x0[0]- sigma, x0[0]+sigma);
		x[1] = rnd.Rannyu(x0[1]- sigma, x0[1]+sigma);
		x[2] = rnd.Rannyu(x0[2]- sigma, x0[2]+sigma);

	}else
	{
		cerr<<"No such distribution--using uniform distribution as default"<<endl;
		x[0] = rnd.Rannyu(x0[0]- sigma, x0[0]+sigma);
		x[1] = rnd.Rannyu(x0[1]- sigma, x0[1]+sigma);
		x[2] = rnd.Rannyu(x0[2]- sigma, x0[2]+sigma);		
	}

	double alpha =p(x)/p(x0);//min(1., p(x)/p(x0)); //both distribution are simmetric
	//accept/ reject

	double r = rnd.Rannyu();


	if (r<=alpha)
	{
		acceptance_rate.push_back(acceptance_rate.back()+1);	
		return x;
	}else
	{
		acceptance_rate.push_back(acceptance_rate.back());	
 		return x0;
	}


}

void read_input(double &sigma, int &nblocks, int &nthrows)
{
	ifstream input("input.dat");

	string property;

	while(!input.eof())
	{
		input >> property;
		if(property == "SIGMA")
		{
			input >> sigma;

		}
		else if (property=="NBLOCKS") 
		{
		 	input>>nblocks;
		}
		else if (property == "NTHROWS") {
			input >> nthrows;
		}
		else {
			cerr << "UNEXPECTED INPUT\n";
			exit(1);
		}
	}

}


void compute_orbital_ray(
	prob p, //orbital probability function

	vector<double> x0, //initial pos for metropolis

	
	//output parameters
	string position_file="",  //file to store positions generated from metropolis
	string ray_values_file="", //file to store the estimated mean value of r

	// metropolis parameters
	string initial_distribution="gauss",
	double sigma=0.5, //initial distribution width parameter 

	//data blocking parameters
	int nblocks=100,
	int nthrows=1000000

)
{
	Random rnd;

	//init seed
	set_seed(rnd);
	int nthr=nthrows/nblocks; //throws per block 
	
	vector<double> avg(nblocks, 0.);
	vector<double> av2(nblocks, 0.);

	vector<double> sum_prog(nblocks, 0.);
	vector<double> su2_prog(nblocks, 0.);
	vector<double> err     (nblocks, 0.);

	//vector of throws
	vector<int> thr(nblocks);
	iota(thr.begin(), thr.end(), 0);
	vector<double> x(x0);
	vector <float> acceptance(1, 0.);
	ofstream out("data/"+position_file);
	
	if(!out.is_open())
	{
		cerr<<"ERROR::IMPOSSIBLE TO OPEN data/"<<position_file<<endl;
		return ;
	}
	out<<"step,x,y,z"<<endl;	

	//for on blocks
	for (int N=0; N<nblocks; ++N)
	{
		double sum=0;
		for(int j=0; j<nthr; ++j)
		{
			x = metropolis(x, p, rnd, sigma, acceptance, initial_distribution); 
			
			//print points on file
			out<<j<<","<<x[0]<<","<<x[1]<<","<<x[2]<<endl;

			sum+=norm(x);
		}
		avg[N]=sum/nthr;
		av2[N]=avg[N]*avg[N];
		
		
		//sum on precedent blocks
		for(int k=0; k<=N; ++k)
		{
			sum_prog[N]+=avg[k];
			su2_prog[N]+=av2[k];
		}
		

		
		sum_prog[N]/=(N+1);
		su2_prog[N]/=(N+1);


		err[N]=error(sum_prog, su2_prog, N);			

		progress(N, nblocks, 40);

	}
	cout<<endl<<endl;		
	out.close();
	for(int &el: thr )
	{
		el*=nthr;
	}

	//save data
	string datafile="data/"+ray_values_file;
	save_data(thr, sum_prog, err, datafile);

	//save acceptance
	vector <int> t(nthrows+1);
	std::iota(t.begin(), t.end(), 0);
	save_data(t, acceptance, "data/acceptance_"+position_file);


	
	
}

int main(int argc, char* argv[])
{
	vector<double> x0(3, 0.); //origin
	
	
	int nblocks=100;
	int nthrows=1000000;
	
	// double sigma;
	// read_input(sigma, nblocks, nthrows);

	cout<<"\n\t\t--> Computing 1s orbital <--\n"<<endl;
	compute_orbital_ray(
		p_1s, x0, "pos_1s_unif.csv", "ray_1s_unif.csv", 
		"uniform", 
		1.2, nblocks, nthrows
	);
	compute_orbital_ray(
		p_1s, x0, "pos_1s_gauss.csv", "ray_1s_gauss.csv", 
		"gauss", 
		0.8, nblocks, nthrows
	);

	cout<<"\n\t\t--> Computing 2p m=0 orbital <--\n"<<endl;
	x0={1., 1., 1.};
	compute_orbital_ray(
		p_2p, x0, "pos_2p_unif.csv", "ray_2p_unif.csv", 
		"uniform", 
		3.0, nblocks, nthrows
	);
	compute_orbital_ray(
		p_2p, x0, "pos_2p_gauss.csv", "ray_2p_gauss.csv", 
		"gauss", 
		1.8, nblocks, nthrows
	);
	
	
	// //other distributions

	// cout<<"\n\t\t--> Computing 2p m=1 orbital <--\n"<<endl;
	// x0={1., 1., 1.};
	// compute_orbital_ray(
	// 	p_2pm1, x0, "pos_2p_m1_gauss.csv", "ray_2p_m1_gauss.csv", 
	// 	"gauss", sigma, nblocks, nthrows
	// );

	// cout<<"\n\t\t--> Computing 2s orbital <--\n"<<endl;
	// x0={0., 0., 0.};
	// compute_orbital_ray(
	// 	p_2s, x0, "pos_2s_gauss.csv", "ray_2s_gauss.csv", 
	// 	"gauss", sigma, nblocks, nthrows
	// );

	// cout<<"\n\t\t--> Computing 3s orbital <--\n"<<endl;
	// x0={0., 0., 1.};
	// compute_orbital_ray(
	// 	p_3s, x0, "pos_3s_gauss.csv", "ray_3s_gauss.csv", 
	// 	"gauss", sigma, nblocks, nthrows
	// );

	// cout<<"\n\t\t--> Computing 3p m=0 orbital <--\n"<<endl;
	// x0={1., 1., 1.};
	// compute_orbital_ray(
	// 	p_3pm0, x0, "pos_3p_m0_gauss.csv", "ray_3p_m0_gauss.csv", 
	// 	"gauss", sigma, nblocks, nthrows
	// );

	// cout<<"\n\t\t--> Computing 3d m=0 orbital <--\n"<<endl;
	// x0={1., 1., 1.};
	// compute_orbital_ray(
	// 	p_3dm0, x0, "pos_3d_m0_gauss.csv", "ray_3d_m0_gauss.csv", 
	// 	"gauss", sigma, nblocks, nthrows
	// );
	
	return 0;
}
