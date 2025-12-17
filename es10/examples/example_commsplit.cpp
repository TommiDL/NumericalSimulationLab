#include <iostream>
#include <mpich-x86_64/mpi.h>
#include <mpich-x86_64/mpi_proto.h>

using namespace std;

int main(int argc, char*argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(size!=4)
    {
        cout << "Servono 4 processi non "<< size<<endl;
        exit(1);
    }

    int icolor, ikey;
    
    if(rank==0){icolor=1;ikey=2;}
    if(rank==1){icolor=1;ikey=1;}
    if(rank==2){icolor=2;ikey=1;}
    if(rank==3){icolor=2;ikey=2;}

    MPI_Comm newcomm;
    MPI_Comm_split(
        MPI_COMM_WORLD,
        icolor, ikey, 
        &newcomm
    );

    int newsize, newrank;
    MPI_Comm_size(newcomm, &newsize);
    MPI_Comm_rank(newcomm, &newrank);

    cout << "ero: "
        <<rank<<" di "<<size
        <<" e ora sono " << newrank << " di " << newsize
        << endl;
    

    return 0;
}