
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

#define BX 16
#define BY 16

// Shared-memory tiled stencil kernel
__global__ void stencilShared2D(double *input,
                                double *output, int N) {

    // Shared memory including halo cells
    __shared__ double tile[BY+2][BX+2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BX + tx;
    int y = blockIdx.y * BY + ty;

    // Load central tile cells
    if(x < N && y < N)
        tile[ty+1][tx+1] = input[y*N+x];
    else
        tile[ty+1][tx+1] = 20.0;

    // Load left halo
    if(tx == 0) {
        tile[ty+1][0] =
            (x > 0 && y < N) ? input[y*N+x-1] : 20.0;
    }

    // Load right halo
    if(tx == BX-1) {
        tile[ty+1][BX+1] =
            (x+1 < N && y < N) ? input[y*N+x+1] : 20.0;
    }

    // Load top halo
    if(ty == 0) {
        tile[0][tx+1] =
            (y > 0 && x < N) ? input[(y-1)*N+x] : 20.0;
    }

    // Load bottom halo
    if(ty == BY-1) {
        tile[BY+1][tx+1] =
            (y+1 < N && x < N) ? input[(y+1)*N+x] : 20.0;
    }

    // Wait until all threads finish loading
    __syncthreads();

    // Calculate output
    if(x < N && y < N) {

        int id = y*N+x;

        // Preserve boundary cells
        if(x == 0 || y == 0 || x == N-1 || y == N-1) {
            output[id] = input[id];
        }
        else {
            output[id] = (
                tile[ty][tx+1] +
                tile[ty+2][tx+1] +
                tile[ty+1][tx] +
                tile[ty+1][tx+2]
            ) / 4.0;
        }
    }
}

int main() {

    const int N = 16384;
    const int ITERATIONS = 100;

    size_t bytes = (size_t)N * N * sizeof(double);

    // Initialize grid
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
                    B[id] = (
                        A[id-N] + A[id+N] +
                        A[id-1] + A[id+1]
                    ) / 4.0;
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

    // Configure 16x16 threads per block
    dim3 block(BX, BY);
    dim3 grid((N+BX-1)/BX, (N+BY-1)/BY);

    // GPU timing
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for(int iter = 0; iter < ITERATIONS; iter++) {

        stencilShared2D<<<grid, block>>>(dA, dB, N);

        CHECK(cudaGetLastError());

        swap(dA, dB);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;

    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy GPU result to CPU
    vector<double> GPU(N*N);

    CHECK(cudaMemcpy(GPU.data(), dA, bytes,
                     cudaMemcpyDeviceToHost));

    // Verify against CPU
    double maxError = 0.0;

    for(int i = 0; i < N*N; i++) {
        maxError = max(maxError, abs(A[i]-GPU[i]));
    }

    cout << fixed << setprecision(8);

    cout << "\nShared Memory: " << BX << " x " << BY;
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
