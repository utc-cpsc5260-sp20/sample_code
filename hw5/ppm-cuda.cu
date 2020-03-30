// nvcc -Xcompiler -Wall -DDOLOG ppm-cuda.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>

// https://stackoverflow.com/questions/28896001/read-write-to-ppm-image-file-c

#ifdef DOLOG
#define LOG(msg) std::cerr<<msg<<std::endl
//#define LOG(msg) fprintf(stderr, msg "\n");
#else
#define LOG(msg)
#endif

// host code for validating last cuda operation (not kernel launch)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



char* data;

int read(std::string filename,
         int& width,
         int& height,
         std::vector<float>& r,
         std::vector<float>& g,
         std::vector<float>& b)
{
    std::ifstream in(filename.c_str(), std::ios::binary);

    int maxcol;

    if (! in.is_open())
    {
        std::cerr << "could not open " << filename << " for reading" << std::endl;
        return 0;
    }

    {
        std::string magicNum;
        in >> magicNum;
        LOG("got magicNum:" << magicNum);

        // this is broken if magicNum != 'P6'
    }

    {
        long loc = in.tellg();
        std::string comment;
        in >> comment;

        if (comment[0] != '#')
        {
            in.seekg(loc);
        }
        else
        {
            LOG("got comment:" << comment);
        }
    }

    in >> width >> height >> maxcol;
    in.get();                   // eat newline
    LOG("dimensions: " << width << "x" << height << "("<<maxcol<<")");
    

//    char* data = new char[width*height*3];
    data = new char[width*height*3];
    in.read(data, width*height*3);
    in.close();
    
    r.resize(width*height);
    g.resize(width*height);
    b.resize(width*height);

    for (int i=0; i<width*height; ++i)
    {
        int base = i*3;
        r[i] =  ((unsigned char)data[base+0])/255.0f;
        g[i] =  ((unsigned char)data[base+1])/255.0f;
        b[i] =  ((unsigned char)data[base+2])/255.0f;
    }
    free(data);

    return 1;
}


int write(std::string outfile,
          int width, int height,
          const std::vector<float>& r,
          const std::vector<float>& g,
          const std::vector<float>& b)
{
    std::ofstream ofs(outfile.c_str(), std::ios::out | std::ios::binary);

    if (! ofs.is_open())
    {
        std::cerr << "could not open " << outfile << " for writing" << std::endl;
    }

    ofs << "P6\n#*\n" << width << " " << height << "\n255\n";

    for (int i=0; i < width*height; ++i)
    {
        ofs <<
            (unsigned char)(r[i]*255) <<
            (unsigned char)(g[i]*255) <<
            (unsigned char)(b[i]*255);
    }
    ofs.close();
    
    return 1;
}



#define imin(a,b) (a<b?a:b)

int main(int argc, char *argv[])
{
    int width, height;

    std::vector<float> r,g,b;
    float *d_r, *d_g, *d_b;

#ifdef DO_READ

    read("input.ppm", width, height, r,g,b);
    LOG("processing " << width << "x" << height);


    // copy read image to GPU
    gpuErrchk(cudaMalloc(&d_r, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, width*height*sizeof(float)));

    gpuErrchk(cudaMemcpy(d_r, &r[0], width*height*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_g, &g[0], width*height*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, &b[0], width*height*sizeof(float), cudaMemcpyHostToDevice));
#else

    width=640;
    height=480;

    gpuErrchk(cudaMalloc(&d_r, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, width*height*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, width*height*sizeof(float)));

#endif
    

    
    // call kernel

    dim3 tpb(16, 16);
    //dim3 tpb(1, 1);
    dim3 bpg((width+tpb.x-1)/tpb.x, (height+tpb.y-1)/tpb.y);

    // for example ....
    // process<<<bpg,tpb>>>(width, height, d_r, d_g, d_b);


    // copy data back from kernel
    r.resize(width*height);
    g.resize(width*height);
    b.resize(width*height);
    
    gpuErrchk(cudaMemcpy(&r[0], d_r, width*height*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&g[0], d_g, width*height*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&b[0], d_b, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
   
    // save image
    write("output.ppm", width, height, r,g,b);

    return 0;
}