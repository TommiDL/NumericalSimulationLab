#include "map.h"
#include "population.h"
#include <string>
#include <mpich-x86_64/mpi.h>

using namespace std;


int main(int argc, char* argv[])
{

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;
    MPI_Request req;
    coordinates coord;//(coord_file);

    coord.initialize();

    if(rank == 0) coord.print();
    // cout << "---end of coord --\n";

    // initialize seed of every process with his rank
    int seed [4] {0,0,0, rank};
    std::srand(rank);
    population pop(coord, seed);
    //pop.print();


    pop.set_avg_filename(
        "avg_cost_rank_"+to_string(rank)
        +".dat"
    );
    pop.set_best_filename(
        "best_sol_rank_"+to_string(rank)
        +".dat"
    );

    pop.initialize();
 
    if(rank == 0) 
    {
        pop.print_setup();
        string msg = (pop.get_migration())?"true":"false";
        cout << "Migration " << msg << endl;
    }// cout << "Evolution started\n";

    crom immigrant(coord.get_ncities(), 0);

    for(int j=0; j<pop.get_nsteps(); j++)
    {

        if((j%pop.get_nmigr()==0) and pop.get_migration())
        // if(false)
        {
            crom bs = pop.get_best();
            
            MPI_Isend(
                &bs[0], bs.size(), 
                MPI_INTEGER, (rank+1)%size, 
                rank, MPI_COMM_WORLD, &req
            );

            int src = (rank==0)?size-1:rank-1;
            
            int newnum=0;
            MPI_Recv(
                &immigrant[0], 
                immigrant.size(), 
                MPI_INTEGER, src, 
                src, MPI_COMM_WORLD, 
                &stat
            );
            // cout << "Processor "<<rank <<"executed it's" << j/pop.get_nmigr()<<"immigration"<<endl;

            MPI_Wait(&req, &stat);
            pop.receive_migration(immigrant);

        }

        pop.evolve();
        pop.save_population_best();
        pop.save_avg_cost();
        // progress(j, pop.get_nsteps());

    }
    // cout<<RESET<<endl;

    //pop.print();
    cout << "Evolution ended"<<endl;

    MPI_Finalize();

    return 0;

}