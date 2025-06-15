#include <iostream>
#include "../es1/func.h"
#include <cmath>

/*
This function provide a prevision of the asset price at time t 
following a GBM evolution with
	- S0 initial asset price
	- mu drift value
	- sigma volatility  
*/
double S(
	double t, 
	double S0, double mu, double sigma, 
	Random &rnd
)
{
	double W=rnd.Gauss(0, t);
	
	return S0*( exp((mu-0.5*sigma*sigma)*t + sigma*W ) ) ;
}


int main(int argc, char* argv[])
{
	//set initial parameters
	double t=0.; 	//initial time
	double S0=100;  //initial price
	double T=1;	//delivery time
	double K=100;	//strike price
	double r=0.1;	//risk-free interest rate
	double sig=0.25;//volatility
	
	int M=10000;
	
	int nblock=100;
	int L=M/nblock;
	
	Random rnd;
	
	set_seed(rnd);
	
	//average on block for strike price
	vector<double> avg(nblock, 0.);
	vector<double> av2(nblock, 0.);
	

	//progressive sum for strike price
	vector<double> sum(nblock, 0.);
	vector<double> su2(nblock, 0.);

	//average on block for call price
	vector<double> cal(nblock, 0.);
	vector<double> ca2(nblock, 0.);
	//progressive sum for call price
	vector<double> cal_sum(nblock, 0.);
	vector<double> ca2_sum(nblock, 0.);

	//average on block for put price
	vector<double> put(nblock, 0.);
	vector<double> pu2(nblock, 0.);
	//progressive sum for put price
	vector<double> put_sum(nblock, 0.);
	vector<double> pu2_sum(nblock, 0.);


	//error on call  and put price
	vector<double> err_cal(nblock, 0.);
	vector<double> err_put(nblock, 0.);
	//error on strike price
	vector<double> err(nblock, 0.);
	
	

	vector<int> thr(nblock, 0.);
	iota(thr.begin(), thr.end(), 0);


	for(int block=0; block<nblock; ++block)
	{
		for(int l=0; l<L; ++l)
		{
			//strike price MC
			double St=S(T, S0, r, sig, rnd);

			//compute strike price average
			avg[block]+=St;

			//compute call price average
			cal[block]+=max(0., St-K)*exp(-r*T);
			//compute put price average
			put[block]+=max(0., K-St)*exp(-r*T);


		}
		avg[block]/=L;
		av2[block]=avg[block]*avg[block];

		cal[block]/=L;
		ca2[block]=cal[block]*cal[block];
		
		put[block]/=L;
		pu2[block]=put[block]*put[block];

		
		for(int j=0; j<=block; ++j)
		{
			sum[block]+=avg[j];
			su2[block]+=av2[j];
			
			cal_sum[block]+=cal[j];
			ca2_sum[block]+=ca2[j];
			
			put_sum[block]+=put[j];
			pu2_sum[block]+=pu2[j];
			
		}
		sum[block]/=(block+1);
		su2[block]/=(block+1);
		err[block]=error(sum, su2, block);
		
		cal_sum[block]/=(block+1);
		ca2_sum[block]/=(block+1);
		err_cal[block]=error(cal_sum, ca2_sum, block);

		put_sum[block]/=(block+1);
		pu2_sum[block]/=(block+1);
		err_put[block]=error(put_sum, pu2_sum, block);

		thr[block]*=L;
	}
	
	save_data(thr, sum, err, "data/strike_priceT.csv");
	save_data(thr, cal_sum, err_cal, "data/call_priceT.csv");	
	save_data(thr, put_sum, err_put, "data/put_priceT.csv");	
	
	
	
	return 0;
}
