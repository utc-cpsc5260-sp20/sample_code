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

#pragma omp parallel for default(none) shared(array) reduction(+:sum)
    for (int i=0; i<SIZE; i++)
    {
        sum += array[i];
    }
    
    double time_e = omp_get_wtime();
    printf("sum: %d\n", sum);
    printf("took %lf sec on %d threads\n", (time_e-time_s),
           omp_get_max_threads());

    return 0;
}
