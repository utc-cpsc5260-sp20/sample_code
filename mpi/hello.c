#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, np;

    int namelen;
    char name[MPI_MAX_PROCESSOR_NAME];


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    MPI_Get_processor_name(name, &namelen);

    printf("rank %d / %d -- %s\n", rank, np, name);
    

    MPI_Finalize();
    return 0;
}
