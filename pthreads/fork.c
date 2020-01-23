#include <stdio.h>
#include <unistd.h>

/* This was just a demonstration of a non-threaded way to write multi-process
 * programs in Unix */

int main(int argc, char *argv[])
{
    printf("hello\n");

    if (fork() == 0)
    {
        sleep(1);
        printf("the child\n");
    }
    else
    {
        printf("the parent\n");
    }
    return 0;
}
