/* Note: In msg2_demo we used Isend/Irecv to avoid deadlock behavior.  As was
   discussed during class, mismatched blocking Send/Recv calls can lead to
   deadlock, though my sample code did not demonstrate this behavior.

   This occurred because the MPI Standard allows MPI to use a "ready" or
   "eager" send mechanism when messages are small (in other words, MPI_Send
   was not blocking!)  This code replaces MPI_Send with a function that
   implementings a synchronous/blocking send regardless of the size of the
   message, namely MPI_Ssend.

   You'll see by experimenting that mismtached Send/Receive calls do indeed
   cause the deadlock error we expect.  Please note that MPI_Send uses this
   same synchronized/blocking logic whenever the message size is large enough
   (which is fairly small for real applications).  I forgot about this when I
   began my demonstration because my research codes do not behave this way,
   since the messages are nontrivial.

*/




/* To compile a  version that deadlocks:
  mpicc -Wall hang_demo.c
 */


/* To compile a working version:
  mpicc -Wall -DDONT_HANG hang_demo.c
 */



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

        // send 1 int to rank=1
        MPI_Ssend(&send_msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf ("%d -- sent message\n", rank);


        MPI_Recv(&recv_msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf ("%d -- received message %d\n", rank, recv_msg);
    }
    else                        /* rank must be 1 since np=2 */
    {
        int send_msg=1111111;
        int recv_msg=0;

#ifndef DONT_HANG
        /* this ordering causes deadlock since both are sending waiting for
         * the other to receive */


        MPI_Ssend(&send_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf ("%d -- sent message\n", rank);

        MPI_Recv(&recv_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf ("%d -- received message %d\n", rank, recv_msg);

#else
        /* this one works as expected */

        MPI_Recv(&recv_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf ("%d -- received message %d\n", rank, recv_msg);

        MPI_Ssend(&send_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf ("%d -- sent message\n", rank);

#endif
        


    }



    MPI_Finalize();
    return 0;
}
