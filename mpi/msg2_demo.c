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


    if (rank == 0)
    {
        int send_msg=9995823;
        int recv_msg=0;

        MPI_Request request[2];


        // send 1 int to rank=1
        MPI_Isend(&send_msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                  &request[0]);
        printf ("%d -- sent message\n", rank);


        MPI_Irecv(&recv_msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                  &request[1]);
        printf ("%d -- received message %d\n", rank, recv_msg);


        // calculate??

        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        // now it's safe to access/modify message data
        printf ("%d -- received message %d\n", rank, recv_msg);

    }
    else                        /* rank must be 1 since np=2 */
    {
        int send_msg=1111111;
        int recv_msg=0;

        MPI_Request request[2];

        MPI_Isend(&send_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                  &request[0]);
        printf ("%d -- sent message\n", rank);

        MPI_Irecv(&recv_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                  &request[1]);
        printf ("%d -- received message %d\n", rank, recv_msg);

        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        printf ("%d -- received message %d\n", rank, recv_msg);


    }



    MPI_Finalize();
    return 0;
}
