#include <stdio.h>
#include <pthread.h>


pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

void* t1(void* x)
{
    pthread_mutex_lock(&lock1);
    printf("in t1\n");
    pthread_mutex_lock(&lock2);
    pthread_mutex_unlock(&lock1);
    pthread_mutex_unlock(&lock2);
    return NULL;
}

/* i should have used trylock! */

void* t2(void* x)
{
    pthread_mutex_lock(&lock2);
    printf("in t2\n");
    pthread_mutex_lock(&lock1);
    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);
    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t th1, th2;

    pthread_create(&th1, NULL, t1, NULL);
    pthread_create(&th2, NULL, t2, NULL);

    pthread_join(th1, NULL);
    pthread_join(th2, NULL);
    return 0;
}
