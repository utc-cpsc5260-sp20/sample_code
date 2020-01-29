#include <stdio.h>
#include <pthread.h>

int sum=0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* threadfunc(void* arg)
{
    // sum = sum+1
    pthread_mutex_lock(&lock);
    sum++;
    pthread_mutex_unlock(&lock);

    return NULL;
}

#define NUM_THREADS 1000

int main(int argc, char *argv[])
{
    pthread_t tids[NUM_THREADS];

    for (int i =0; i<NUM_THREADS; i++)
    {
        pthread_create(&tids[i], NULL, threadfunc, NULL);
    }
    for (int i =0; i<NUM_THREADS; i++)
    {
        pthread_join(tids[i], NULL);
    }

    printf("sum: %d\n", sum);

    return 0;
}
