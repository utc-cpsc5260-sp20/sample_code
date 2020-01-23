/* build with  gcc -Wall -O3 -pthread pthread1.c */
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 10

/* This variable is available to every thread, but unsynchronized read/write
 * access to the variable will lead to problems. */
int global=0;


/* Pthreads entry point */
void* threadfunc(void* arg)
{
    /* follow the arg pointer to the integer passed in.  This is a flexible
     * approach in that it allows for arbitrarily complex argument data to be
     * used */
    int rank=*(int*)arg;

    /* alternately, the value could be packed into the void* 
    int rank=(int)(long int)arg;
    */

    /* Dangerous accesses of global ...*/
    global++;
    printf("thread %d -- %d\n", rank, global);
    sleep(1);


    /* Either one of these approaches will return `rank + 100' to the spawning
     * thread, retrievable in pthread_join.

     It works by squeezing the numeric value into the 8-byte void* For more
     complex return data, you should think about dynamic memory management or
     a return struct passed in by the caller.
     */


    return (void*)(long int)(rank+100);
    /*
      use pthread_exit when threads call multiple subroutines and returning all
      the way is impractical

      pthread_exit( (void*)(long int)(rank+100) );
      return NULL;
    */
    

}

int main(int argc, char *argv[])
{
    printf("at start\n");

    /* Stack-allocated thread id's and individual thread arguments */
    pthread_t tid[NUM_THREADS];
    int args[NUM_THREADS];

    for (int i=0; i< NUM_THREADS; i++)
    {
        args[i]=i;
        pthread_create(&tid[i], NULL, threadfunc, &args[i]);
        
        /* alternately, stuff that into into the void* ...
           pthread_create(&tid[i], NULL, threadfunc, (void*)(long int)i);
        */
    }
    printf("waiting for children\n");


    for (int i=0; i< NUM_THREADS; i++)
    {
        void* ret;
        pthread_join(tid[i], &ret);
        int answer=(int)(long int)ret;
        printf("thread %d returned %d\n", i, answer);

        //        pthread_cancel to end threads before they're finished
    }


    // pthread_join(tid, NULL);
    printf("after join\n");


    return 0;
}
