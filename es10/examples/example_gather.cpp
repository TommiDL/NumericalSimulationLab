#include <iostream>
#include <mpich-x86_64/mpi.h>

using namespace std;

int main(int argc, char*argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);


    if(size>3)
    {
        cout << "Hai scelto troppi processi"<<endl;
        return 1; 
    }

    int irecv[3];
    for(int i=0; i<3;i++)
    {
        irecv[i]=0;
    }
    int isend = rank+1; // send to the next

    MPI_Gather(
        &isend, 1, MPI_INTEGER,  // send isend 
         irecv, 1, MPI_INTEGER, //receive it into irecv
         0, MPI_COMM_WORLD      // handled from process 0
    );

    if(rank==0)
        cout << "IRECV: "
            << irecv[0]<<" "
            << irecv[1]<<" "
            << irecv[2]
            << endl;
    
    MPI_Finalize();

    return 0;
}