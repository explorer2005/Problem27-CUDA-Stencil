
#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from CUDA Thread %d\n", threadIdx.x);
}

int main() {

    helloCUDA<<<1, 5>>>();

    cudaDeviceSynchronize();

    return 0;
}
