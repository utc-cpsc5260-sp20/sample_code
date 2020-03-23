#include <stdio.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define imin(a,b) (a<b?a:b)


const int N=33000*1024;

const int tpb = 256;            // threads per block
const int bpg = imin(32, (N+tpb-1) / tpb);


// this kernel computes a local sum by calculating the product of certain
// element pairs and maintaining a local sum of the results, which are
// subsequently reduced.

__global__ void dot(float *a, float* b, float* c)
{
    __shared__ float cache[tpb];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp=0;
    
    // each thread is responsible for values in the original arrays, strided
    // by blockDim*gridDim.  Neighboring threads in a warp will request
    // adjacent array elements, leading to better memory access performance
    // (coalesced reads)
    while (tid < N) 
    {
        // c[tid]=a[tid]*b[tid];
        temp += a[tid]*b[tid];
        tid += blockDim.x * gridDim.x;
    }


    // Now reduce the sum on each thread block


    // the shared memory has room for one value per thread.  
    cache[threadIdx.x] = temp;

    // wait for all threads to finish the above while loop and reduce ...
    __syncthreads();
    int cutoff = blockDim.x /2 ;

    while (cutoff != 0)
    {
        if (threadIdx.x < cutoff)
        {
            cache[threadIdx.x] += cache[threadIdx.x + cutoff];
        }

        __syncthreads();
        cutoff /= 2;
    }

    // cache[0] contains the reduced value for this thread block
    if (threadIdx.x == 0)
    {
        c[blockIdx.x] = cache[0];
    }
    
}

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(bpg*sizeof(float));

    gpuErrchk(cudaMalloc(&d_a, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_c, bpg*sizeof(float)));
    
    for (int i=0; i<N; i++) 
    {
        a[i] = i;
        b[i] = i*2;
    }

    gpuErrchk(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    
    cudaEvent_t start, stop;
    float elapsed;

    // this allows GPU-level timing of the kernel (elapsed time between kernel
    // launch and subsequent synchronized call
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    dot<<<bpg, tpb>>>(d_a, d_b, d_c);
    gpuErrchk( cudaDeviceSynchronize() );

    
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("cuda kernel: %.2f ms\n", elapsed);
        
    
    // here's the "boring" host-based reduction of the threaded code above.
    // Notice this could be done entirely on the GPU with a second kernel that
    // just did a reduction.
    gpuErrchk(cudaMemcpy(c, d_c, bpg*sizeof(float), cudaMemcpyDeviceToHost));

    float temp=0;
    for (int i=0; i<bpg; i++)
    {
        temp += c[i];
    }

    // this manual solution should equal the closed form solution for these vectors
    printf ("solved: %.6g\nclosed: %.6g\n", temp, 2*sum_squares((float)(N-1)));


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}