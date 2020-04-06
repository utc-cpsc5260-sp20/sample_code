#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, np;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np != 2)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }


    int msg=0;

    if (rank == 0)
    {
        msg=9995823;

        // send 1 int to rank=1
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf ("%d -- sent message\n", rank);
        

    }
    else                        /* rank must be 1 since np=2 */
    {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf ("%d -- received message %d\n", rank, msg);
    }



    MPI_Finalize();
    return 0;
}
