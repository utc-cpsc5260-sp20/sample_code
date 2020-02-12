#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define SIZE 1000000

int main(int argc, char *argv[])
{
    int* array=(int*)malloc(SIZE*sizeof(int));
    int sum=0;

    for (int i=0; i<SIZE; i++)
    {
        array[i]=i;
        sum += array[i];
    }
    printf("real answer: %d\n", sum);

    sum=0;

    double time_s = omp_get_wtime();

#pragma omp parallel default(none) shared(array, sum)
    {
        int rank=omp_get_thread_num();
        int np=omp_get_num_threads();
        int count=SIZE/np;
        int start=rank*count;
        int mod=SIZE%np;

        int local_sum=0;

        if (rank < mod)
        {
            count++;
            start+=rank;
        }
        else
        {
            start+=mod;
        }

        for(int i=start; i<start+count; i++)
        {
            /* here's a big critical section */
            /*
            #pragma omp critical
            {
                sum += array[i];
            }
            */
            /* or this..             #pragma omp atomic */

            local_sum += array[i];
        }

        #pragma omp atomic
        sum += local_sum;
        
    }
    double time_e = omp_get_wtime();
    printf("sum: %d\n", sum);
    printf("took %lf sec on %d threads\n", (time_e-time_s),
           omp_get_max_threads());

    return 0;
}
