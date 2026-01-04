# CUDA-2D-Convolution-Optimization

This project implements a CUDA-based 2D convolution kernel and progressively optimizes it by improving memory access patterns, exploiting on-chip memory, and applying register-level optimizations. The goal is to study how GPU memory hierarchy and execution models affect performance.

## Key Ideas

### Thread–Data Remapping
Redesigned thread indexing and block geometry to ensure that each warp accesses contiguous memory locations, enabling fully coalesced global memory loads and reducing fragmented DRAM transactions.

### Explicit Memory Hierarchy Management
Used shared memory tiling to reuse input data across threads and constant memory for read-only filter weights, significantly reducing global memory traffic and improving effective bandwidth utilization.

### Register Tiling and Thread Coarsening
Increased per-thread work by computing multiple output elements per thread and reusing values already loaded into registers, exposing instruction-level parallelism and reducing memory pressure.

## Results

- **4.2×** improvement in effective DRAM bandwidth due to coalesced global memory accesses  
- **6.5×** increase in kernel throughput from shared and constant memory optimizations  
- **15×** end-to-end throughput improvement with a **94% reduction in kernel runtime**

Performance analysis and profiling were conducted using **NVIDIA Nsight Compute**.

## Repository Structure

- `conv_kernels.cu`: CUDA kernels and optimized variants  
- `report.pdf`: Detailed analysis and profiling results
