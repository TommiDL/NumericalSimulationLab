/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Prof. D.E. Galli
_/    _/  _/_/_/  _/_/_/_/ email: Davide.Galli@unimi.it
*****************************************************************
*****************************************************************/

#include <iostream>
#include "system.h"

using namespace std;

void progress(int block, int nblock, int barlenght=30)
{
	int l=barlenght*float(block)/nblock;
	string str="[";
	for (int b=0; b<l; b++) str+="|";
	cout<<"\r"<<str<<setw(barlenght-l+1)<<"]["<<100*l/float(barlenght)<<"%]"<<flush;
} 

int main (int argc, char *argv[]){

  int nconf = 1;
  System SYS;
  SYS.initialize();
  SYS.initialize_properties();
  SYS.block_reset(0);
  int nb=SYS.get_nbl();
  //cout<<"sym started"<<endl;
  //float perc=0;
  cout<<"Simulation Started"<<endl;
  cout<<"["<<setw(20)<<"]";
    
  for(int i=0; i < SYS.get_nbl(); i++){ //loop over blocks

    for(int j=0; j < SYS.get_nsteps(); j++){ //loop over steps in a block
      SYS.step();
      SYS.measure();
      if(j%50 == 0){
        SYS.write_XYZ(nconf); //Write actual configuration in XYZ format //Commented to avoid "filesystem full"! 
        nconf++;
      }
    }
    SYS.averages(i+1);
    SYS.block_reset(i+1);
    //perc=i*100./nb; //update perc
    progress(i, nb);

  }
  cout<<endl<<endl;
  SYS.finalize();
  return 0;
}

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Prof. D.E. Galli
_/    _/  _/_/_/  _/_/_/_/ email: Davide.Galli@unimi.it
*****************************************************************
*****************************************************************/
