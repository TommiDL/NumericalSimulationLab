#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>
#include "func.h"

using namespace std;

int main(int argc, char* argv[])
{
    double d=1.0, L=0.8;
    int Nthr=10000;
    int nblock=100;

    int th = Nthr/nblock;

    Random rnd;
    set_seed(rnd);

    vector<int> throws(nblock); 
	iota(throws.begin(), throws.end(), 0);

    vector<double> sum_prog(nblock, 0.);
    vector<double> su2_prog(nblock, 0.);
    vector<double> err_prog(nblock, 0.);


    double sum=0,sum2=0;
    for(int k=0; k< nblock; k++)
    {
        int success=0;

        double x0, x1, x_dir,y_dir,proj;
        for(int j=0; j<th; j++)
        {
            // plane has simmetry of d for x traslation
            x0 = rnd.Rannyu(0, d);

            x_dir=rnd.Rannyu(-1, 1);
            y_dir=rnd.Rannyu(-1, 1);

            proj = x_dir/sqrt(x_dir*x_dir+y_dir*y_dir);

            if(x0==0)
            {
                //intersection in the first point
                success+=1;
            }

            x1= x0 + L*proj;
            if ((x1>=d) || (x1<=0))
            {
                //intersection
                success+=1;
            }
        
        }

        double pi = 2*L*(th)/(success*d);
        sum += pi;
        sum2+= pi*pi;

        sum_prog[k] = sum/(k+1);
        su2_prog[k] = sum2/(k+1);
        err_prog[k] =  error(sum_prog, su2_prog, k);

    }



    for (auto & element : throws)
        element*=th; //number of throw = block * (number of throw for every block)

    save_data(throws, sum_prog, err_prog, "data/prob_buffon.csv");

    return 0;
}