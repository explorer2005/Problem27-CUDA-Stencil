#include <chrono>

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
using namespace std;

#define IDX(z,y,x,N) ((z)*(N)*(N) + (y)*(N) + (x))
#define TILE 16

// 2.5D Stencil: Shared XY tile + Z register pipeline
__global__ void stencil3D_25D_register(
    double *A, double *B, int N)
{
    __shared__ double tile[TILE+2][TILE+2];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    bool valid = (x < N && y < N);

    // Each thread keeps three Z-plane values in registers
    double previous = 0.0;
    double current  = 0.0;
    double next     = 0.0;

    if (valid)
    {
        previous = A[IDX(0,y,x,N)];
        current  = A[IDX(1,y,x,N)];
        next     = A[IDX(2,y,x,N)];

        // Preserve first and last Z boundary planes
        B[IDX(0,y,x,N)] = previous;
        B[IDX(N-1,y,x,N)] = A[IDX(N-1,y,x,N)];
    }

    // Slide through the Z dimension
    for (int z = 1; z < N-1; z++)
    {
        // Load current Z plane into shared memory
        if (valid)
            tile[ty+1][tx+1] = current;

        // Load left and right halo
        if (valid && tx == 0 && x > 0)
            tile[ty+1][0] =
                A[IDX(z,y,x-1,N)];

        if (valid && tx == TILE-1 && x < N-1)
            tile[ty+1][TILE+1] =
                A[IDX(z,y,x+1,N)];

        // Load top and bottom halo
        if (valid && ty == 0 && y > 0)
            tile[0][tx+1] =
                A[IDX(z,y-1,x,N)];

        if (valid && ty == TILE-1 && y < N-1)
            tile[TILE+1][tx+1] =
                A[IDX(z,y+1,x,N)];

        // Synchronize shared-memory tile
        __syncthreads();

        if (valid)
        {
            int i = IDX(z,y,x,N);

            // Preserve X and Y boundaries
            if (x == 0 || y == 0 ||
                x == N-1 || y == N-1)
            {
                B[i] = current;
            }
            else
            {
                // Four XY neighbours from shared memory
                double left  = tile[ty+1][tx];
                double right = tile[ty+1][tx+2];
                double up    = tile[ty][tx+1];
                double down  = tile[ty+2][tx+1];

                // Two Z neighbours from registers
                B[i] = (left + right + up + down
                        + previous + next) / 6.0;
            }
        }

        // All threads finish reading shared memory
        // before it is overwritten in the next iteration
        __syncthreads();

        // Advance the register pipeline
        previous = current;
        current = next;

        // Load only the next required Z plane
        if (valid && z+2 < N)
            next = A[IDX(z+2,y,x,N)];
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
    const int N = 256;
    const int ITER = 5;

    size_t total = (size_t)N * N * N;
    size_t bytes = total * sizeof(double);

    double *h_A = new double[total];
    double *h_B = new double[total];
    double *h_ref = new double[total];
    double *temp = new double[total];

    // Initialize grid
    for (int z = 0; z < N; z++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                int i = IDX(z,y,x,N);

                if (x == 0 || y == 0 || z == 0 ||
                    x == N-1 || y == N-1 || z == N-1)
                {
                    h_A[i] = 20.0;
                }
                else
                {
                    h_A[i] = 30.0
                           + (x % 7) * 0.1
                           + (y % 5) * 0.1
                           + (z % 3) * 0.1;
                }
            }
        }
    }

    copy(h_A, h_A + total, h_ref);

    double *d_A, *d_B;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(
        h2d_end - h2d_start).count();
    

    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // GPU iterations: grid remains on device
    for (int iter = 0; iter < ITER; iter++)
    {
        stencil3D_25D_register<<<grid, block>>>(
            d_A, d_B, N
        );

        double *swapPtr = d_A;
        d_A = d_B;
        d_B = swapPtr;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds, start, stop
    );

    
    auto d2h_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_B, d_A, bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::high_resolution_clock::now();
    double d2h_ms = std::chrono::duration<double, std::milli>(
        d2h_end - d2h_start).count();
    

    // CPU reference iterations
    for (int iter = 0; iter < ITER; iter++)
    {
        stencilCPU(h_ref, temp, N);
        swap(h_ref, temp);
    }

    // Compare CPU and GPU results
    double maxError = 0.0;

    for (size_t i = 0; i < total; i++)
    {
        maxError = max(
            maxError,
            fabs(h_B[i] - h_ref[i])
        );
    }

    cout << "3D 2.5D Register Pipeline" << endl;
    cout << "Grid Size: " << N << " x "
         << N << " x " << N << endl;

    cout << "Iterations: " << ITER << endl;

    cout << "GPU Kernel Time: "
         << milliseconds << " ms" << endl;

    cout << "Maximum Absolute Error: "
         << maxError << endl;

    if (maxError < 1e-6)
        cout << "VERIFICATION PASSED!" << endl;
    else
        cout << "VERIFICATION FAILED!" << endl;

    
    cout << "H2D Transfer Time: " << h2d_ms << " ms" << endl;
    cout << "D2H Transfer Time: " << d2h_ms << " ms" << endl;

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
