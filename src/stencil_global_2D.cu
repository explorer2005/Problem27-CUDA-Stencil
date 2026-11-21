
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        cerr << cudaGetErrorString(err) << endl; \
        return 1; \
    } \
} while(0)

// One GPU thread calculates one grid cell
__global__ void stencil2D(double *input, double *output, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= N || y >= N) return;

    int id = y * N + x;

    // Keep boundary cells unchanged
    if(x == 0 || y == 0 || x == N-1 || y == N-1) {
        output[id] = input[id];
    }
    else {
        output[id] = (
            input[id-N] +
            input[id+N] +
            input[id-1] +
            input[id+1]
        ) / 4.0;
    }
}

int main() {

    const int N = 16384;
    const int ITERATIONS = 100;

    size_t bytes = N * N * sizeof(double);

    // Initialize input grid
    vector<double> initial(N*N, 20.0);

    for(int y = 1; y < N-1; y++) {
        for(int x = 1; x < N-1; x++) {
            initial[y*N+x] = 30.0;
        }
    }

    initial[(N/2)*N + N/2] = 100.0;

    // CPU reference
    vector<double> A = initial;
    vector<double> B(N*N);

    for(int iter = 0; iter < ITERATIONS; iter++) {

        for(int y = 0; y < N; y++) {
            for(int x = 0; x < N; x++) {

                int id = y*N+x;

                if(x == 0 || y == 0 || x == N-1 || y == N-1)
                    B[id] = A[id];
                else
                    B[id] = (A[id-N] + A[id+N] +
                             A[id-1] + A[id+1]) / 4.0;
            }
        }

        A.swap(B);
    }

    // Allocate GPU memory
    double *dA, *dB;

    CHECK(cudaMalloc((void**)&dA, bytes));
    CHECK(cudaMalloc((void**)&dB, bytes));

    CHECK(cudaMemcpy(dA, initial.data(), bytes,
                     cudaMemcpyHostToDevice));

    // Configure CUDA threads and blocks
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (N+15)/16);

    // Measure GPU kernel execution time
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < ITERATIONS; iter++) {

        stencil2D<<<grid, block>>>(dA, dB, N);

        CHECK(cudaGetLastError());

        swap(dA, dB);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;

    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy final GPU result to CPU
    vector<double> GPU(N*N);

    CHECK(cudaMemcpy(GPU.data(), dA, bytes,
                     cudaMemcpyDeviceToHost));

    // Verify GPU against CPU
    double maxError = 0.0;

    for(int i = 0; i < N*N; i++) {
        maxError = max(maxError, abs(A[i] - GPU[i]));
    }

    cout << fixed << setprecision(8);

    cout << "\nGrid Size: " << N << " x " << N;
    cout << "\nIterations: " << ITERATIONS;
    cout << "\nGPU Kernel Time: " << gpuTime << " ms";
    cout << "\nMaximum Absolute Error: " << maxError;

    if(maxError < 1e-6)
        cout << "\nVERIFICATION PASSED!\n";
    else
        cout << "\nVERIFICATION FAILED!\n";

    cudaFree(dA);
    cudaFree(dB);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
