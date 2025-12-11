#include <iostream>
#include "random_walk.h"
#include <fstream>

int main(int argc, char* argv[])
{	
	
	//save the steps of the different realizations of RW
	ofstream out("data/RW_lattice.csv"); 
	
	//save the mean value and variance of quadratic distance computed using data blocking
	//on the different realization of RWs
	ofstream stat_out("data/RW_rn.csv");
	if(!out.is_open())
	{
		cerr<<"ERROR: cannot open RW_lattice.csv"<<endl;
		return -1;	
	}
	if(!stat_out.is_open())
	{
		cerr<<"ERROR: cannot open RW_rn.csv"<<endl;
		return -1;	
	}
		
	
	//number of realizations
	int N=10000;
	
	//number of steps
	int M=100;
	
	//data block dimension
	int nblock=100;
	int L=N/nblock;
	
	
	out<<"#rw,step,x,y,z"<<endl;
	RW_lattice rw(1.);
	
	//2d vectors of shape (Nblocks, step)
	vector<vector<double>> avg(M, vector<double> (nblock, 0.));
	vector<vector<double>> av2(M, vector<double> (nblock, 0.));
	
	
	
//	for(int run=0; run<N; ++run)
	for(int block=0; block<nblock; ++block)
	{
		for(int l=0; l<L; ++l) // for each block make L RWs
		{	
				
			int run=L*block+l;
			
			/*****************       new RW          ******************/

			vector<double> point = rw.get_pos();
			// save initial step
			out <<run<<","<<rw.get_nstep()<<","
				<<point[0]<<","<<point[1]<<","<<point[2]<<endl;
			

			for (int k=0; k<M; ++k) //for on steps
			{
				rw.make_step();
				
				// save step
				point=rw.get_pos();
				out<<run<<","<<rw.get_nstep()
					<<","<<point[0]<<","<<point[1]<<","<<point[2]<<endl;
				
				//for each step calculate the average on nblock walks
				avg[k][block]+=sqrt(
					 point[0]*point[0]
					+point[1]*point[1]
					+point[2]*point[2]
				);
				
				
			}
			
			rw.reset_rw(); // reset the RW in pos 0 and set step to 0
		}
		
		//performe mean and square mean
		for(int k=0; k<M; ++k)
		{
			avg[k][block]/=L;
			av2[k][block]=avg[k][block]*avg[k][block];
		}
	}	
	
	//save the RWs
	out.close();


	//init file
	stat_out<<"N,block,rN,err"<<endl;
	
	vector<vector<double>> sum(M, vector<double> (nblock, 0.));
	vector<vector<double>> su2(M, vector<double> (nblock, 0.));
	vector<vector<double>> err(M, vector<double> (nblock, 0.));
	// cumulative mean over blocks
	for(int k=0; k<M; ++k) //step iteration
	{
		for(int block=0; block<nblock; ++block)
		{
			for(int j=0; j<=block; ++j)
			{
				sum[k][block]+=avg[k][j];
				su2[k][block]+=av2[k][j];
			}
			sum[k][block]/=(block+1);
			su2[k][block]/=(block+1);
			err[k][block]=error(sum[k], su2[k], block);
		
			stat_out<<k<<","<<block<<","
				<<sum[k][block]<<","<<err[k][block]<<endl;		
		
		}

	}
	
	stat_out.close();
	
	
	return 0;
}
