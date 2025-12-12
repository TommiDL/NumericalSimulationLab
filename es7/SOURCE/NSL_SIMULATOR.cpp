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
#include "../../NSL_SIMULATOR/SOURCE/system.h"
#include "../../NSL_SIMULATOR/SOURCE/function.h"

using namespace std;


int main (int argc, char *argv[]){

  int nconf = 1;
  System SYS;
  SYS.initialize();
  SYS.initialize_properties();
  //cout<<"Simulating ising at temperature "<<BLUE <<tmin+ts*step_temp <<endl;
  SYS.block_reset(0);
  int nb=SYS.get_nbl();

  
  SYS.print_details();
  SYS.equilibration_steps();
  cout << "Symulation started"<<endl;

  for(int i=0; i < SYS.get_nbl(); i++){ //loop over blocks
    for(int j=0; j < SYS.get_nsteps(); j++){ //loop over steps in a block
      SYS.step();
      SYS.measure();
      if(j%50 == 0){
        //SYS.write_XYZ(nconf); //Write actual configuration in XYZ format //Commented to avoid "filesystem full"! 
        nconf++;
      }
    }
    SYS.averages(i+1);
    
    SYS.block_reset(i+1);

    progress(i, nb);

  }

  cout<<RESET<<endl<<endl;
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
