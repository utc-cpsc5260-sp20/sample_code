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


const int N=33*1024;
const int tpb = 256;            // threads per block
const int bpg = imin(32, (N+tpb-1) / tpb);

__global__ void dot(float *a, float* b, float* c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) 
    {
        c[tid]=a[tid]*b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));

    gpuErrchk(cudaMalloc(&d_a, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_c, N*sizeof(float)));
    
    for (int i=0; i<N; i++) 
    {
        a[i] = i;
        b[i] = i*2;
    }

    gpuErrchk(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    dot<<<bpg, tpb>>>(d_a, d_b, d_c);

    gpuErrchk(cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    float temp=0;
    for (int i=0; i<N; i++)
    {
        temp += c[i];
    }

    printf ("solved: %.6g\nclosed: %.6g\n", temp, 2*sum_squares((float)(N-1)));


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}