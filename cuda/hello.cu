// nvcc -Xcompiler -Wall -arch=sm_35 hello.cu -o hello

#include "cuda_helpers.h"


__global__ void hello()
{
    /* printf("hello from the gpu\n"); */
    printf("thread %d,%d,%d/%d,%d,%d - block: %d,%d,%d/%d,%d,%d\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z
        );
}


int main(int argc, char *argv[])
{
     hello<<<3,5>>>(); 
     /* hello<<<1,dim3(2,3,4)>>>();  */

    gpuErrchk(cudaDeviceSynchronize());

    return 0;
}