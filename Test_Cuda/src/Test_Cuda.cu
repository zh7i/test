//============================================================================
// Name        : Test_Cuda.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description : Test of the segment add plug-in
//============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define PER_THREAD_INIT_NUM 8
#define PER_THREAD_SEG_NUM 16
#define ARRAY_SIZE (1024 * 1024 * 128)
#define MAX_NUM 256

template <typename T>
struct SumOp {
    __device__ void operator()(T *addr, T value) {
        atomicAdd(addr, value);
    }
};

template <typename T>
struct MaxOp {
    __device__ void operator()(T *addr, T value) {
        atomicAdd(addr, value);
    }
};

__global__ void InitValue(int *input, int total_size, int value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int split_size = total_size / PER_THREAD_INIT_NUM;
#pragma unroll
    for (int i = 0; i < PER_THREAD_INIT_NUM; i++) {
        input[i * split_size + index] = value;
    }
}

template <typename ReductionF>
__global__ void Segmentation(int *__restrict__ input_data, int *__restrict__ input_segment_id, int *__restrict__ output_segment_id, int segment_id_size, int total_size) {

    extern __shared__ int segment_result_slm[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int split_size = total_size / PER_THREAD_SEG_NUM;

    if (threadIdx.x < segment_id_size) {
        segment_result_slm[threadIdx.x] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < PER_THREAD_SEG_NUM; i++) {
        int segment_id = input_segment_id[split_size * i + index];
        int data_value = input_data[split_size * i + index];
        atomicAdd(segment_result_slm + segment_id, data_value);
    }
    __syncthreads();

    if (threadIdx.x < segment_id_size) {
        ReductionF()(output_segment_id + threadIdx.x, segment_result_slm[threadIdx.x]);
    }
}

int main(int argc, char **argv) {
    int *input_data = nullptr;
    int *input_segment_id = nullptr;
    int *output_segment_id = nullptr;

    int *input_data_cpu = nullptr;
    int *input_segment_id_cpu = nullptr;
    int *output_segment_id_cpu = nullptr;
    int *output_segment_id_gold = nullptr;
    int array_num = ARRAY_SIZE;
    int segment_num = 512;
    int thread_num = 512;
    float time = 0.0f;

    cudaEvent_t event1, event2;

    std::srand(std::time(nullptr));
    int dev = cudaSetDevice(0);

    CUDA_CHECK_RETURN(cudaMalloc((void **)&input_data, array_num * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&input_segment_id, array_num * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&output_segment_id, segment_num * sizeof(int)));

    CUDA_CHECK_RETURN(cudaEventCreate(&event1));
    CUDA_CHECK_RETURN(cudaEventCreate(&event2));

    if ((input_data_cpu = new int[array_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((input_segment_id_cpu = new int[array_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((output_segment_id_cpu = new int[segment_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((output_segment_id_gold = new int[segment_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    // Initialize array
    for (int i = 0; i < array_num; i++) {
        input_data_cpu[i] = rand() % MAX_NUM;
    }
    CUDA_CHECK_RETURN(cudaMemcpy(input_data, input_data_cpu, array_num * sizeof(int), cudaMemcpyHostToDevice));

    for (int i = 0; i < array_num; i++) {
        input_segment_id_cpu[i] = rand() % segment_num;
    }
    CUDA_CHECK_RETURN(cudaMemcpy(input_segment_id, input_segment_id_cpu, array_num * sizeof(int), cudaMemcpyHostToDevice));




    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    InitValue<<<1, segment_num / PER_THREAD_INIT_NUM>>>(output_segment_id, segment_num, 0);
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Kernel InitValue Execution time " << time << " ms" << std::endl;


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    Segmentation<SumOp<int>><<<array_num / PER_THREAD_SEG_NUM / (thread_num * 2), (thread_num * 2), segment_num * sizeof(int)>>>(input_data, input_segment_id, output_segment_id, segment_num, array_num);
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));
    std::cout << "Kernel Segmentation Reduction time " << time << " ms" << std::endl;


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    CUDA_CHECK_RETURN(cudaMemcpy(output_segment_id_cpu, output_segment_id, segment_num * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Memory copy time: " << time << " ms" << std::endl;

    // compute and compare gold
    memset(output_segment_id_gold, 0, segment_num * sizeof(int));

    clock_t begin_clock = std::clock();
    for (int i = 0; i < array_num; i++) {
        output_segment_id_gold[input_segment_id_cpu[i]] += input_data_cpu[i];
    }
    clock_t end_clock = std::clock();
    double elapsed_cpu = double(end_clock - begin_clock) / CLOCKS_PER_SEC * 1000.0;
    std::cout << "CPU reduction time " << elapsed_cpu << " ms" << std::endl;

    for (int i = 0; i < segment_num; i++) {
        if (output_segment_id_cpu[i] != output_segment_id_gold[i]) {
            std::cout << "Error comparing with Gold! " << "Position " << i << std::endl;
            return -1;
        }
    }

    std::cout << "Test passed! " << std::endl;

    if (input_data_cpu != nullptr) {
        delete[] input_data_cpu;
    }
    if (input_segment_id_cpu != nullptr) {
        delete[] input_segment_id_cpu;
    }
    if (output_segment_id_cpu != nullptr) {
        delete[] output_segment_id_cpu;
    }
    if (output_segment_id_gold != nullptr) {
        delete[] output_segment_id_gold;
    }

    cudaFree(output_segment_id);
    cudaFree(input_segment_id);
    cudaFree(input_data);
    return 0;
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}



