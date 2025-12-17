#include <iostream>
#include <mpich-x86_64/mpi.h>
#include <mpich-x86_64/mpi_proto.h>

using namespace std;

const int n=100; // try to increase n
// se n troppo grande esaurisco il buffer, si risolve switchando in 0 il send e il receive

int main(int argc, char*argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status stat1, stat2; // per contenere info slla comunicazione

    int* imesg=new int [n];
    int* imesg2=new int [n];
    
    int itag=1; //bandierina da mettere sulla comunicazione
    int itag2=2; //bandierina da mettere sulla comunicazione

    for(int i =0; i<n; i++)
    {
        imesg[i]=rank;
        imesg2[i]=rank+1;
    }

    if(rank==1)
    {
        MPI_Send(
            &imesg[0],n, //mando 0 ma dico di mandarne n alora li manda tutti poi mi metto in ricezione
            MPI_INTEGER, 0, itag, MPI_COMM_WORLD
        );

        MPI_Recv(
            &imesg2[0], n,
            MPI_INTEGER, 0, itag2, MPI_COMM_WORLD, &stat2);

        cout << "messaggio from 0 to 1 = "<< imesg2[0]<<endl;
    }
    else if(rank==0)
    {

        MPI_Recv(
            &imesg[0], n,
            MPI_INTEGER, 1, itag, MPI_COMM_WORLD, &stat1);

        MPI_Send(
            &imesg2[0],n, //mando 0 ma dico di mandarne n alora li manda tutti poi mi metto in ricezione
            MPI_INTEGER, 1, itag2, MPI_COMM_WORLD
        );
        cout << "messaggio from 1 to 0 = "<< imesg[0]<<endl;
    }

    MPI_Finalize();

    return 0;
}