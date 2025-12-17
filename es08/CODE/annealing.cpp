#include "function.h"
#include "system.h"
#include <string>

int main()
{
    System SYS;
    
    SYS.initialize();
    SYS.sample_probability("OUTPUT/initial_wf.dat", 10000);

    int nst=SYS.get_n_annealing_steps();
    while (!SYS.terminated()) 
    {
        string desc="T = " + to_string(SYS.get_temperature());
        for (int k=0; k<nst; k++) 
        {
            SYS.anneal_move();
            progress(k, nst, 30, desc);
            
            // SYS.sample_probability(
            //     "wf_evol/wf_step"
            //     +to_string(k)+".dat"
            // );
        }
        cout << endl;
        SYS.cooling();    
    }
    SYS.set_energy_filename(
        "OUTPUT/final_energy.dat"
    );

    SYS.compute_energy(false, false, true);

    SYS.sample_probability("OUTPUT/final_wf.dat", 10000);
    return 0;
}