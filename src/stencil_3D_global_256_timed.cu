#include <chrono>

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
using namespace std;

#define IDX(z,y,x,N) ((z)*(N)*(N) + (y)*(N) + (x))

// 3D Global Memory Stencil Kernel
__global__ void stencil3D(double *A, double *B, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= N || y >= N || z >= N) return;

    int i = IDX(z,y,x,N);

    if (x == 0 || y == 0 || z == 0 ||
        x == N-1 || y == N-1 || z == N-1)
    {
        B[i] = A[i];
    }
    else
    {
        B[i] = (
            A[IDX(z,y,x-1,N)] + A[IDX(z,y,x+1,N)] +
            A[IDX(z,y-1,x,N)] + A[IDX(z,y+1,x,N)] +
            A[IDX(z-1,y,x,N)] + A[IDX(z+1,y,x,N)]
        ) / 6.0;
    }
}

// CPU Reference
void cpuStencil(double *A, double *B, int N)
{
    for (int z=0; z<N; z++)
        for (int y=0; y<N; y++)
            for (int x=0; x<N; x++)
            {
                int i = IDX(z,y,x,N);

                if (x==0 || y==0 || z==0 ||
                    x==N-1 || y==N-1 || z==N-1)
                    B[i] = A[i];
                else
                    B[i] = (
                        A[IDX(z,y,x-1,N)] + A[IDX(z,y,x+1,N)] +
                        A[IDX(z,y-1,x,N)] + A[IDX(z,y+1,x,N)] +
                        A[IDX(z-1,y,x,N)] + A[IDX(z+1,y,x,N)]
                    ) / 6.0;
            }
}

int main()
{
    const int N = 256;
    const int ITERATIONS = 5;
    size_t total = (size_t)N*N*N;
    size_t bytes = total*sizeof(double);

    double *h_A = new double[total];
    double *h_B = new double[total];
    double *h_ref = new double[total];

    // Initialize grid
    for (int z=0; z<N; z++)
        for (int y=0; y<N; y++)
            for (int x=0; x<N; x++)
            {
                int i = IDX(z,y,x,N);
                h_A[i] = (x==0 || y==0 || z==0 ||
                          x==N-1 || y==N-1 || z==N-1) ? 20.0 : 30.0;
            }

    // Calculate CPU reference
    double *cpu_in = new double[total];
    double *cpu_out = new double[total];

    copy(h_A, h_A+total, cpu_in);

    for (int iter=0; iter<ITERATIONS; iter++)
    {
        cpuStencil(cpu_in, cpu_out, N);
        swap(cpu_in, cpu_out);
    }

    copy(cpu_in, cpu_in+total, h_ref);

    // Allocate GPU memory
    double *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(
        h2d_end - h2d_start).count();
    

    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int iter=0; iter<ITERATIONS; iter++)
    {
        stencil3D<<<grid,block>>>(d_A,d_B,N);
        swap(d_A,d_B);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    
    auto d2h_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_B, d_A, bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::high_resolution_clock::now();
    double d2h_ms = std::chrono::duration<double, std::milli>(
        d2h_end - d2h_start).count();
    

    // Verify GPU result
    double maxError = 0.0;

    for (size_t i=0; i<total; i++)
        maxError = max(maxError, fabs(h_B[i]-h_ref[i]));

    cout << "3D Grid Size: " << N << " x " << N << " x " << N << endl;
    cout << "Iterations: " << ITERATIONS << endl;
    cout << "GPU Kernel Time: " << ms << " ms" << endl;
    cout << "Maximum Absolute Error: " << maxError << endl;

    cout << (maxError < 1e-6 ?
             "VERIFICATION PASSED!" :
             "VERIFICATION FAILED!") << endl;

    
    cout << "H2D Transfer Time: " << h2d_ms << " ms" << endl;
    cout << "D2H Transfer Time: " << d2h_ms << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);

    delete[] h_A;
    delete[] h_B;
    delete[] h_ref;
    delete[] cpu_in;
    delete[] cpu_out;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
