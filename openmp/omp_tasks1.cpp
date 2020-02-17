#include <cstdio>
#include <omp.h>
#include <unistd.h>


int main(int argc, char *argv[])
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i=0; i<10; i++)
            {
                #pragma omp task
                {
                    printf("task %d started - %d\n", i,
                           omp_get_thread_num());
                    sleep(1);
                    printf("task %d ended\n", i);
                }
            }
            

        }


    }

    
    return 0;
}
