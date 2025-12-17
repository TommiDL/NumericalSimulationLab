#include <iostream>
#include <mpich-x86_64/mpi.h>

using namespace std;


int main(int argc, char*argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status stat; // per contenere info slla comunicazione

    double tstart=MPI_Wtime();
    int n=100;
    int* imesg=new int [n];
    int sum=0;

    for(int i =0; i<n; i++)
    {
        imesg[i]=rank;
        if(rank==1)
        {
            MPI_Send(&imesg[0], n, MPI_INTEGER, 0, i, MPI_COMM_WORLD);
        }
        else if(rank==0)
        {
            MPI_Recv(&imesg[0], n, MPI_INTEGER, 1, i, MPI_COMM_WORLD, &stat);
        }

        sum+=imesg[i];
    }

    double tend=MPI_Wtime();
    double dt=tend-tstart;

    cout <<"io sono "<<rank<<", somma="<<sum<<"tempo="<<dt<<endl;

    MPI_Finalize();

    return 0;
}