
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
using namespace std;

#define IDX(z,y,x,N) ((z)*(N)*(N) + (y)*(N) + (x))
#define TILE 16

// Shared-memory tiled 3D stencil kernel
__global__ void stencil3D_25D(double *A, double *B, int N)
{
    __shared__ double tile[TILE+2][TILE+2];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    int z = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    bool valid = (x < N && y < N && z < N);

    int i = IDX(z,y,x,N);

    // Load central tile
    if (valid)
        tile[ty+1][tx+1] = A[i];

    // Load left and right halo cells
    if (valid && tx == 0 && x > 0)
        tile[ty+1][0] = A[IDX(z,y,x-1,N)];

    if (valid && tx == TILE-1 && x < N-1)
        tile[ty+1][TILE+1] = A[IDX(z,y,x+1,N)];

    // Load top and bottom halo cells
    if (valid && ty == 0 && y > 0)
        tile[0][tx+1] = A[IDX(z,y-1,x,N)];

    if (valid && ty == TILE-1 && y < N-1)
        tile[TILE+1][tx+1] = A[IDX(z,y+1,x,N)];

    // Synchronize all threads before using shared memory
    __syncthreads();

    // Calculate output only for valid threads
    if (valid)
    {
        // Preserve boundary cells
        if (x == 0 || y == 0 || z == 0 ||
            x == N-1 || y == N-1 || z == N-1)
        {
            B[i] = A[i];
        }
        else
        {
            // Four neighbours from shared memory
            double left  = tile[ty+1][tx];
            double right = tile[ty+1][tx+2];
            double up    = tile[ty][tx+1];
            double down  = tile[ty+2][tx+1];

            // Two neighbours from global memory along Z
            double front = A[IDX(z-1,y,x,N)];
            double back  = A[IDX(z+1,y,x,N)];

            // Six-neighbour average
            B[i] = (left + right + up + down + front + back) / 6.0;
        }
    }
}

// CPU reference implementation
void stencilCPU(double *A, double *B, int N)
{
    for (int z = 0; z < N; z++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                int i = IDX(z,y,x,N);

                // Preserve boundaries
                if (x == 0 || y == 0 || z == 0 ||
                    x == N-1 || y == N-1 || z == N-1)
                {
                    B[i] = A[i];
                }
                else
                {
                    B[i] = (
                        A[IDX(z,y,x-1,N)] +
                        A[IDX(z,y,x+1,N)] +
                        A[IDX(z,y-1,x,N)] +
                        A[IDX(z,y+1,x,N)] +
                        A[IDX(z-1,y,x,N)] +
                        A[IDX(z+1,y,x,N)]
                    ) / 6.0;
                }
            }
        }
    }
}

int main()
{
    const int N = 512;
    const int ITER = 5;

    size_t bytes = (size_t)N * N * N * sizeof(double);

    double *h_A = new double[N*N*N];
    double *h_B = new double[N*N*N];
    double *h_ref = new double[N*N*N];

    // Initialize input grid
    for (int z = 0; z < N; z++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                int i = IDX(z,y,x,N);

                if (x == 0 || y == 0 || z == 0 ||
                    x == N-1 || y == N-1 || z == N-1)
                    h_A[i] = 20.0;
                else
                    h_A[i] = 30.0 + (x % 7) * 0.1
                                   + (y % 5) * 0.1
                                   + (z % 3) * 0.1;
            }
        }
    }

    copy(h_A, h_A + N*N*N, h_ref);

    double *d_A, *d_B;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE,
              (N+TILE-1)/TILE, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // GPU iterations
    for (int iter = 0; iter < ITER; iter++)
    {
        stencil3D_25D<<<grid, block>>>(d_A, d_B, N);

        double *temp = d_A;
        d_A = d_B;
        d_B = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy final GPU result
    cudaMemcpy(h_B, d_A, bytes, cudaMemcpyDeviceToHost);

    // CPU reference iterations
    double *temp = new double[N*N*N];

    for (int iter = 0; iter < ITER; iter++)
    {
        stencilCPU(h_ref, temp, N);
        swap(h_ref, temp);
    }

    // Compare CPU and GPU results
    double maxError = 0.0;

    for (int i = 0; i < N*N*N; i++)
    {
        maxError = max(maxError, fabs(h_B[i] - h_ref[i]));
    }

    cout << "3D Shared Memory Grid Size: "
         << N << " x " << N << " x " << N << endl;

    cout << "Iterations: " << ITER << endl;

    cout << "GPU Kernel Time: "
         << milliseconds << " ms" << endl;

    cout << "Maximum Absolute Error: "
         << maxError << endl;

    if (maxError < 1e-6)
        cout << "VERIFICATION PASSED!" << endl;
    else
        cout << "VERIFICATION FAILED!" << endl;

    cudaFree(d_A);
    cudaFree(d_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_A;
    delete[] h_B;
    delete[] h_ref;
    delete[] temp;

    return 0;
}
