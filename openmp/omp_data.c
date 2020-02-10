#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int x=-1;

int main(int argc, char *argv[])
{
/*
     static  int x=-1;
*/

#pragma omp  threadprivate(x)


#pragma omp parallel default(none)
    {
        x = omp_get_thread_num();
        sleep(1);
        printf("block 1: %d - %d\n", omp_get_thread_num(), x);
    }

#pragma omp parallel default(none)
    {
        //x = omp_get_thread_num();
        sleep(1);
        printf("block 2: %d - %d\n", omp_get_thread_num(), x);
    }

/*
#pragma omp parallel default(none) private(x)
    {
        x = omp_get_thread_num();
        sleep(1);
        printf("block 1: %d - %d\n", omp_get_thread_num(), x);
    }

#pragma omp parallel default(none) private(x)
    {
        //x = omp_get_thread_num();
        sleep(1);
        printf("block 2: %d - %d\n", omp_get_thread_num(), x);
    }
*/

    printf("back serial: %d\n", x);


    return 0;
}
