# CUDA-Accelerated Stencil Computation

## Overview
This project implements and benchmarks 2D and 3D stencil computations
using CUDA. It compares global memory, shared memory, and a 2.5D
register-based pipeline.

## Implementations
- Sequential CPU 2D stencil
- CUDA 2D global memory stencil
- CUDA 2D shared memory stencil
- CUDA 3D global memory stencil
- CUDA 3D shared memory stencil
- CUDA 3D 2.5D register pipeline
- Benchmark variants for 256^3 and 512^3 grids

## Repository Structure
- `src/` - CUDA and C++ source code
- `results/` - CSV benchmark datasets
- `plots/` - Performance comparison graphs
- `report/` - Project report and supporting documents

## Benchmark Environment
- Google Colab
- NVIDIA Tesla T4 GPU
- CUDA 13.0
- Compute capability: 7.5

## Verification
The benchmark implementations were checked against CPU reference
results. The recorded 3D benchmark configurations passed verification
with zero maximum absolute error.

## Benchmark Summary

| Grid | Implementation | Kernel time (ms) |
|---|---|---:|
| 256^3 | Global memory | 16.7257 |
| 256^3 | Shared memory | 15.2708 |
| 256^3 | 2.5D register | 19.8255 |
| 512^3 | Global memory | 156.707 |
| 512^3 | Shared memory | 133.761 |
| 512^3 | 2.5D register | 118.933 |

All above measurements use 5 iterations.

## How to Compile
Example:

```bash
nvcc src/stencil_3D_global_256.cu -o stencil
./stencil
```

Use an NVIDIA GPU with a compatible CUDA toolkit.

## Results
See the `results/` directory for CSV datasets and the `plots/`
directory for performance visualizations.

## Author
Dhruv Gupta
