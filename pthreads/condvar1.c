#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define MAX_COUNT 10
int count=0;

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


void* wait(void* x)
{
    pthread_mutex_lock(&lock);

    while (count < MAX_COUNT)
    {
        printf("waiting...\n");
        pthread_cond_wait(&cond, &lock);
    }

    pthread_mutex_unlock(&lock);
    printf("count: %d\n", count);

    return NULL;
}

void* act(void* x)
{
    while (1)
    {
        sleep(1);
        pthread_mutex_lock(&lock);
        count++;

        if (count >= MAX_COUNT)
        {
            pthread_cond_signal(&cond);
        }

        pthread_mutex_unlock(&lock);

        printf("count: %d\n", count);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t waiter;
    pthread_t actors[NUM_THREADS];

    pthread_create(&waiter, NULL, wait, NULL);
    for (int i=0; i<NUM_THREADS; i++)
    {
        pthread_create(&actors[i], NULL, act, NULL);
    }

    pthread_join(waiter, NULL);
    for (int i=0; i<NUM_THREADS; i++)
    {
        pthread_cancel(actors[i]);
        pthread_join(actors[i], NULL);
    }
    printf("the end\n");

    return 0;


}


