/*
Name : Rishikesh S
Srno : 27076
*/


#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../include/conv_kernels.cuh"
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>

// Defining some constants for block width, height and max possible kernel radius 
#define BLOCK_W 32
#define BLOCK_H 16
#define MAX_RADIUS 7
#define MAX_KERNEL_WIDTH (2*MAX_RADIUS + 1)
#define NUM_STREAMS 4

__constant__ float constant_kernel[MAX_KERNEL_WIDTH*MAX_KERNEL_WIDTH];

// =============================================================================
// BASELINE IMPLEMENTATION (PROVIDED - DO NOT MODIFY)
// =============================================================================
// This is an intentionally inefficient implementation for comparison purposes.

static __device__ __forceinline__ size_t idx3(int batch_index, int row, int col,
                                               int height, int width) {
    return static_cast<size_t>(batch_index) * height * width +
           static_cast<size_t>(row) * width +
           col;
}

__global__ void kernel_conv2d_baseline(const float* __restrict__ input_images,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output_images,
                                       int batch_size, int height, int width,
                                       int kernel_size) {
    int batch_index = blockIdx.z;

    int row = blockIdx.y * blockDim.y + threadIdx.x;  // Note: threadIdx.x for row
    int col = blockIdx.x * blockDim.x + threadIdx.y;  // Note: threadIdx.y for col

    if (batch_index >= batch_size || row >= height || col >= width) {
        return;
    }

    int radius = (kernel_size - 1) / 2;
    float accumulated_value = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
        int input_row = row + kernel_row - radius;
        if (input_row < 0 || input_row >= height) continue;

        for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
            int input_col = col + kernel_col - radius;
            if (input_col < 0 || input_col >= width) continue;

            float input_pixel = input_images[idx3(batch_index, input_row, input_col,
                                                   height, width)];
            float kernel_weight = kernel[kernel_row * kernel_size + kernel_col];
            accumulated_value += input_pixel * kernel_weight;
        }
    }

    output_images[idx3(batch_index, row, col, height, width)] = accumulated_value;
}

void conv2d_baseline(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_baseline<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, kernel, output, batch_size, height, width, kernel_size
    );
}

// =============================================================================
// VARIANT 1: GLOBAL-MEMORY ACCESS PATTERN (BANDWIDTH EFFICIENCY)
// =============================================================================
// TODO: Restructure the thread/data mapping so that the hardware can merge
//       per-thread loads into fewer, fuller memory transactions.
//
// GOAL:
// - Increase effective global-memory bandwidth by maximizing bytes used per
//   transaction and minimizing transactions per warp.
//
// WHAT TO MEASURE (before & after):
// - L1TEX/L2: average bytes used per sector (aim ~32/32 for 32B sectors).
// - Global load efficiency / requested vs. delivered bytes.
// - DRAM read throughput (GB/s) and “transactions per request” counters.
// - Kernel time and MPix/s.
//
// HINTS (discovery-oriented):
// - Inspect how a warp’s threads (lane 0..31) walk the image in memory.
//   Are neighboring lanes reading neighboring addresses, or are they striding?
// - Revisit your mapping from (threadIdx.x, threadIdx.y) → (row, col).
//   Which dimension in memory is contiguous, and do lanes advance along it?
// - Consider block shapes where each warp spans one logical row of the tile
//   rather than splitting a warp across multiple rows.
// - The order of inner loops matters: move the loop that advances along the
//   contiguous memory dimension into the per-lane direction.
// - When alignment permits, loading wider types (e.g., 16-byte aligned chunks)
//   reduces the number of memory transactions. Handle tails safely.
//
// =============================================================================

__global__ void kernel_conv2d_variant1(const float* __restrict__ input_images, 
                                        const float* __restrict__ kernel,
                                        float* __restrict__ output_image,
                                        int batch_size, int height, int width,
                                        int kernel_size) {

    int batch_index = blockIdx.z;

    // XHanging the thread to data mapping for coalesced access 
    int row = (blockIdx.y*blockDim.y) + threadIdx.y ;
    int col = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (batch_index >= batch_size || row >= height || col >= width) {
        return;
    }


    int radius = (kernel_size - 1)/2;
    float accumulated_value = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
        int input_row = (row - radius) + kernel_row;
        if (input_row < 0 || input_row >= height) {
            continue;
        }
        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
            int input_col = (col - radius) + kernel_col;
            if (input_col < 0 || input_col >= width) {
                continue;
            }
            float input_pixel = input_images[idx3(batch_index, input_row, input_col, height, width)];
            float kernel_pixel = kernel[kernel_row*kernel_size + kernel_col];

            accumulated_value += (input_pixel * kernel_pixel);
        }
    }

    output_image[idx3(batch_index, row, col, height, width)] = accumulated_value;
}

void conv2d_variant1(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
   dim3 threads_per_block(BLOCK_W, BLOCK_H, 1);
   dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_variant1<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, kernel, output, batch_size, height, width, kernel_size
    );

}

// =============================================================================
// VARIANT 2: ON-CHIP MEMORY (SHARED + CONSTANT)
// =============================================================================
//
// In this task, you will explore the different levels of the GPU memory hierarchy
// and analyze how they impact performance.
//
// Begin by profiling the naive convolution implementation using NVIDIA Nsight Compute.
// Record key metrics such as memory bandwidth utilization, cache hit rates, IPC, and
// other relevant performance indicators.
//
// Next, study the various GPU memory types — both on-chip and off-chip — and discuss
// their access latencies and bandwidths. Explain which of these memories are being
// used by the naive convolution kernel and how.
//
// Then, implement Variant 2 by modifying the kernel to make use of different on-chip
// memory spaces. Specifically, explore the use of shared memory and constant memory
// to improve data reuse and reduce global memory traffic.
//
// After your optimization, re-profile the kernel and report changes in cache
// utilization, bandwidth utilization, and overall performance.
//
// Finally, observe and explain an interesting phenomenon: certain optimizations may
// increase memory bandwidth utilization while decreasing cache hit rates, yet still
// lead to better performance. Provide a detailed reasoning for why this happens,
// relating it to reduced cache dependence, more efficient data reuse, and improved
// throughput across the GPU memory hierarchy.
//
// =============================================================================

__global__ void kernel_conv2d_variant2(const float* __restrict__ input_images, 
                                        float* __restrict__ output_images,
                                        int batch_size, int height, int width,
                                        int kernel_size) {
    int batch_index = blockIdx.z;

    int row = (blockIdx.y * BLOCK_H) + threadIdx.y;
    int col = (blockIdx.x * BLOCK_W) + threadIdx.x;


    // Declaring shared memory region. Note that only filters whose radius is upto MAX_RADIUS is supported                                     
    __shared__ float conv_region[BLOCK_H + 2*MAX_RADIUS][BLOCK_W + 2*MAX_RADIUS];

    int radius = (kernel_size - 1)/2;
    int shared_height = BLOCK_H + 2*radius;
    int shared_width = BLOCK_W + 2*radius;

    int block_row = blockIdx.y*BLOCK_H;
    int block_col = blockIdx.x*BLOCK_W;

    int sx = threadIdx.x;
    int sy = threadIdx.y;

    // Loading the region into shared memory. We load this in a tiled fashion, which ensures that there are no bank conflicts while accessing shared memory
    // (threadblock width is 32), and the global memory accesses are coalesced (while accessing the input region for loading)
    for (int ty = sy; ty < shared_height; ty += BLOCK_H) {
        int global_row = (block_row - radius) + ty;
        for (int tx = sx; tx < shared_width; tx += BLOCK_W) {
            int global_col = (block_col - radius) + tx;
            float val = 0.0f;
            if (global_row >= 0 && global_row < height && global_col >= 0 && global_col < width) {
                val = input_images[idx3(batch_index, global_row, global_col, height, width)];
            }
            conv_region[ty][tx] = val;
        }
    }

    // To ensure all threads have completed loading their data
    __syncthreads();

    float accumulated_value = 0.0f;

    for (int k_row = 0; k_row < kernel_size; k_row++) {
        for (int k_col = 0; k_col < kernel_size; k_col++) {
            // Reading the kernel filter value from constant memory, and input image from shared memory. 
            float kernel_val = constant_kernel[k_row*kernel_size + k_col];
            float pixel_val = conv_region[threadIdx.y + k_row][threadIdx.x + k_col];
            accumulated_value += kernel_val*pixel_val;
        }
    }

    if (row < height && col < width) {
        output_images[idx3(batch_index, row, col, height, width)] = accumulated_value;
    }

}
void conv2d_variant2(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {

    // Your code here
    dim3 threads_per_block(BLOCK_W, BLOCK_H, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    // Allocation of shared memory 
    size_t kernel_bytes = static_cast<size_t>(kernel_size) * kernel_size * sizeof(float);
    cudaMemcpyToSymbol(constant_kernel, kernel, kernel_bytes, 0, cudaMemcpyDeviceToDevice);

    kernel_conv2d_variant2<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, output, batch_size, height, width, kernel_size
    );

}

// =============================================================================
// VARIANT 3: REGISTER-LEVEL OPTIMIZATION AND DATA LOCALITY
// =============================================================================
//
// In this task, you will investigate the role of the GPU’s register file and
// how exploiting data locality at the thread level can further improve
// performance beyond what shared and global memory optimizations achieve.
//
// Begin by profiling your previous variant and examine metrics related to
// register utilization, instruction-level parallelism (ILP), and arithmetic
// efficiency. Observe how many registers are used per thread and whether
// memory operations still dominate execution time.
//
// Next, study how the GPU register file serves as the fastest storage resource
// available to each thread. Think about ways to reuse data already loaded into
// registers to reduce redundant memory accesses and improve computational
// intensity. Consider whether each thread could perform more useful work by
// computing multiple nearby output elements rather than just one.
//
// Modify the kernel to take advantage of this thread-level reuse and the
// available registers. After your optimization, re-profile and report changes
// in achieved FLOP/s, register utilization, and memory bandwidth usage.
//
// Finally, discuss in your report how locality within the register file and
// the reuse of data across computations can reduce memory pressure and improve
// throughput. Relate your findings to the GPU’s execution model and to the
// balance between register usage, occupancy, and ILP.
//
// =============================================================================

__global__ void kernel_conv2d_variant3(const float* __restrict__ input_images,
                                        float* __restrict__ output_image,
                                        int batch_size, int height, int width,
                                        int kernel_size) {
    int batch_index = blockIdx.z;
    // Change in mapping from previous, as we have halved the number of threadblocks we launch (in both dimensions). This is because now each thread 
    // computes a 2x2 output region.     
    int row = blockIdx.y*2*BLOCK_H + 2*threadIdx.y;
    int col = blockIdx.x*2*BLOCK_W + 2*threadIdx.x;
    if (batch_index >= batch_size) {
        return;
    }

    __shared__ float conv_region[2*BLOCK_H + 2*MAX_RADIUS][2*BLOCK_W + 2*MAX_RADIUS];

    int radius = (kernel_size - 1)/2;
    int shared_height = 2*BLOCK_H + 2*radius;
    int shared_width = 2*BLOCK_W + 2*radius;

    int block_row = blockIdx.y*2*BLOCK_H;
    int block_col = blockIdx.x*2*BLOCK_W;

    int sx = threadIdx.x;
    int sy = threadIdx.y;


    for (int ty = sy; ty < shared_height; ty += BLOCK_H) {
        int global_row = (block_row - radius) + ty;
        for (int tx = sx; tx < shared_width; tx += BLOCK_W) {
            int global_col = (block_col - radius) + tx;
            float val = 0.0f;
            if (global_row >= 0 && global_row < height && global_col >= 0 && global_col < width) {
                val = input_images[idx3(batch_index, global_row, global_col, height, width)];
            }
            conv_region[ty][tx] = val;
        }
    }

    __syncthreads();

    // Each variable accumulates the value for a particular pixel of the 2x2 region 
    float accumulated_value1 = 0.0f;
    float accumulated_value2 = 0.0f;
    float accumulated_value3 = 0.0f;
    float accumulated_value4 = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
        int shared_row1 = 2*threadIdx.y + kernel_row;
        int shared_row2 = shared_row1 + 1;
        int shared_col = 2*threadIdx.x;
        float top_left = conv_region[shared_row1][shared_col];
        float bottom_left = conv_region[shared_row2][shared_col];
        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
            float kernel_val = constant_kernel[kernel_row*kernel_size + kernel_col];
            float next_top = conv_region[shared_row1][shared_col+kernel_col+1];
            float next_bottom = conv_region[shared_row2][shared_col+kernel_col+1];

            accumulated_value1 += top_left*kernel_val;
            accumulated_value2 += next_top*kernel_val;
            accumulated_value3 += bottom_left*kernel_val;
            accumulated_value4 += next_bottom*kernel_val;
            
            // We pass the value through registers to the next iteration. Note that this helps to reduce access to shared memory 
            top_left = next_top;
            bottom_left = next_bottom;
        }   
    }


    // Boundary checks before writing to output image 
    if (row < height) {
        if (col < width) {
            output_image[idx3(batch_index, row, col, height, width)] = accumulated_value1;
        }
        if (col + 1 < width) {
            output_image[idx3(batch_index, row, col+1, height, width)] = accumulated_value2;
        }
    }

    if (row + 1 < height) {
        if (col < width) {
            output_image[idx3(batch_index, row+1, col, height, width)] = accumulated_value3;
        }
        if (col + 1 < width) {
            output_image[idx3(batch_index, row+1, col+1, height, width)] = accumulated_value4;
        }
    }

}

void conv2d_variant3(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {

    // Your code here
    dim3 threads_per_block(BLOCK_W, BLOCK_H, 1);
    // Halving the number of threadblocks we launch
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / (threads_per_block.x*2),
        (height + threads_per_block.y - 1) / (threads_per_block.y*2),
        batch_size
    );

    size_t kernel_bytes = static_cast<size_t>(kernel_size) * kernel_size * sizeof(float);
    cudaMemcpyToSymbol(constant_kernel, kernel, kernel_bytes, 0, cudaMemcpyDeviceToDevice);

    kernel_conv2d_variant3<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, output, batch_size, height, width, kernel_size
    );
}

// =============================================================================
// BONUS: MULTI-STREAM CONCURRENT EXECUTION
// =============================================================================
//
// In this task, use Nsight Systems to understand the end-to-end timeline and
// then improve throughput by overlapping independent work with CUDA streams.
//
// GOAL:
// - Reduce idle gaps on the copy and compute engines by overlapping operations
//   (e.g., host to device transfers with kernel execution) across a large batch.
//
// WHAT TO EXAMINE IN NSIGHT SYSTEMS (BEFORE):
// - Are H2D/D2H copies serialized with kernel launches?
// - Do copy engines (C/E) or SMs sit idle between batches?
// - Where are the longest gaps on the timeline (host prep, copies, kernels)?
//
// WHAT TO MEASURE (before & after):
// - End-to-end time per full batch; GPU utilization (%), copy engine utilization.
// - Degree of overlap visible on the NSYS timeline (copies concurrent with kernels).
// - Any change in kernel performance (avoid starving compute with too many small chunks).
//
// =============================================================================

void conv2d_variant4(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) { 
}

// =============================================================================
// BONUS ROUND
// =============================================================================


void conv2d_bonus(const float* input, const float* kernel1, const float* kernel2,
                  float* output, int batch_size, int height, int width,
                  int kernel_size, cudaStream_t stream) {
    // TODO: Implement multi-stream version
    // You can reuse any of your previous kernels

    // Your code here
}