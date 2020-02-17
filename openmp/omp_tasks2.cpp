//  -*- mode: c++; eval: (c-set-offset (quote cpp-macro) 0)-*- 
#include <cstdio>
#include <omp.h>
#include <unistd.h>

void do_task(const char* name, int duration)
{
    printf("task %s started\n", name);
    sleep(duration);
    printf("task %s ended\n", name);
}

int main(int argc, char *argv[])
{
    double start = omp_get_wtime();
    int A, B, C, D, E;

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task depend(out:A)
            do_task("A", 10);

            #pragma omp task depend(out:B)
            do_task("B", 2);

            #pragma omp task depend(out:C)
            do_task("C", 2);

            #pragma omp task depend(inout:B,C,D)
            do_task("D", 5);

            #pragma omp task depend(in:A,D)
            do_task("E", 1);
        }
    }

    double end = omp_get_wtime();
    printf("time: %lf\n", end-start);
    return 0;
}
