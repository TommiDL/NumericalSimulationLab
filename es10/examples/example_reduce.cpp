#include <iostream>
#include <mpich-x86_64/mpi.h>

using namespace std;

int main(int argc, char*argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);



    int isend[2], irecv[2];
    
    for(int i=0; i<2;i++)
    {
        isend[i]=rank+i+1;
    }

    MPI_Reduce(
        &isend[0], &irecv[0],
        1, MPI_INTEGER, 
        MPI_SUM, 0,     // op to perform and process who handle it 
        MPI_COMM_WORLD
    );


    MPI_Reduce(
        &isend[1], &irecv[1],
        1, MPI_INTEGER, 
        MPI_PROD, 0,     // op to perform and process who handle it 
        MPI_COMM_WORLD
    );

    if(rank==0)
        cout << "IRECV: "
            << irecv[0]<<" "
            << irecv[1]
            << endl;
    
    MPI_Finalize();

    return 0;
}