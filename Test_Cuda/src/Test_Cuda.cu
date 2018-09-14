//============================================================================
// Name        : Unordered Reduction.cpp
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
#include <cmath>
#include <limits>
#include <algorithm>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define PER_THREAD_INIT_NUM 8
#define PER_THREAD_SEG_NUM 64
#define ARRAY_SIZE (1024 * 64)
#define MAX_NUM 4

typedef float operand_type;

template<typename T, typename std::enable_if<std::is_same<T, int64_t>::value == false, void>::type* = nullptr>
__device__ void SumOpByType(T *addr, T value)
{
    atomicAdd(addr, value);
}

template<typename T, typename std::enable_if<std::is_same<T, int64_t>::value, void>::type* = nullptr>
__device__ void SumOpByType(T *addr, T value)
{
    unsigned long long int *addr_temp = (unsigned long long int *)addr;
    unsigned long long int value_temp = (unsigned long long int) value;
    atomicAdd(addr_temp, value_temp);
}

template <typename T>
struct SumOp {
    __device__ void operator()(T *addr, T value) {
        SumOpByType(addr, value);
    }
};

template<typename T, typename std::enable_if<std::is_same<T, float>::value, void>::type* = nullptr>
__device__ void MaxOpByType(T *addr, T value)
{
    unsigned int *address = (unsigned int *)addr;
    unsigned int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              __float_as_uint(value > __uint_as_float(value_assumed) ? value : __uint_as_float(value_assumed)));
    } while (value_assumed != value_old);
}

template<typename T, typename std::enable_if<std::is_same<T, double>::value, void>::type* = nullptr>
__device__ void MaxOpByType(T *addr, T value)
{
    unsigned long long int *address = (unsigned long long int *)addr;
    unsigned long long int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              __double_as_longlong(value > __longlong_as_double(value_assumed) ? value : __longlong_as_double(value_assumed)));
    } while (value_assumed != value_old);

}

template<typename T, typename std::enable_if<std::is_same<T, unsigned int>::value || std::is_same<T, unsigned long long int>::value ||
                                          std::is_same<T, int>::value || std::is_same<T, long long int>::value, void>::type* = nullptr>
__device__ void MaxOpByType(T *addr, T value)
{
    atomicMax(addr, value);
}

template<typename T, typename std::enable_if<std::is_same<T, int64_t>::value, void>::type* = nullptr>
__device__ void MaxOpByType(T *addr, T value)
{
    unsigned long long int *addr_temp = (unsigned long long int *)addr;
    unsigned long long int value_temp = (unsigned long long int) value;
    atomicMax(addr_temp, value_temp);
}

template <typename T>
struct MaxOp {
    __device__ void operator()(T *addr, T value) {
        MaxOpByType(addr, value);
    }
};

template<typename T, typename std::enable_if<std::is_same<T, float>::value, void>::type* = nullptr>
__device__ void MulOpByType(T *addr, T value)
{
    unsigned int *address = (unsigned int *)addr;
    unsigned int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              __float_as_uint(value * __uint_as_float(value_assumed)));
    } while (value_assumed != value_old);
}

template<typename T, typename std::enable_if<std::is_same<T, int>::value || std::is_same<T, unsigned int>::value, void>::type* = nullptr>
__device__ void MulOpByType(T *addr, T value)
{
    unsigned int *address = (unsigned int *)addr;
    unsigned int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              (value * value_assumed));
    } while (value_assumed != value_old);
}

template<typename T, typename std::enable_if<std::is_same<T, double>::value, void>::type* = nullptr>
__device__ void MulOpByType(T *addr, T value)
{
    unsigned long long int *address = (unsigned long long int *)addr;
    unsigned long long int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              __double_as_longlong(value * __longlong_as_double(value_assumed)));
    } while (value_assumed != value_old);
}

template<typename T, typename std::enable_if<std::is_same<T, long long int>::value ||
                                             std::is_same<T, unsigned long long int>::value ||
                                             std::is_same<T, int64_t>::value, void>::type* = nullptr>
__device__ void MulOpByType(T *addr, T value)
{
    unsigned long long int *address = (unsigned long long int *)addr;
    unsigned long long int value_old = *address, value_assumed;

    do {
        value_assumed = value_old;
        value_old = atomicCAS(address,
                              value_assumed,
                              ((unsigned long long int)value * value_assumed));
    } while (value_assumed != value_old);
}

template <typename T>
struct MulOp {
    __device__ void operator()(T *addr, T value) {
        MulOpByType(addr, value);
    }
};

template <typename T>
__global__ void InitValue(T *input, int total_size, T value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int split_size = total_size / PER_THREAD_INIT_NUM;
#pragma unroll
    for (int i = 0; i < PER_THREAD_INIT_NUM; i++) {
        input[i * split_size + index] = value;
    }
}

template <typename ReductionF, typename T>
__global__ void Segmentation(T *__restrict__ input_data,
                             int *__restrict__ input_segment_id,
                             T *__restrict__ output_segment_result,
                             int segment_result_size,
                             int total_size) {

    extern __shared__ int segment_result_slm_shared[];
    T *segment_result_slm = (T *)segment_result_slm_shared;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int split_size = total_size / PER_THREAD_SEG_NUM;

    if (threadIdx.x < segment_result_size) {
        segment_result_slm[threadIdx.x] = 1;
    }
    __syncthreads();

//#pragma unroll
    for (int i = 0; i < PER_THREAD_SEG_NUM; i++) {
        int segment_id = input_segment_id[split_size * i + index];
        T data_value = input_data[split_size * i + index];
        ReductionF()(segment_result_slm + segment_id, data_value);
    }
    __syncthreads();

    if (threadIdx.x < segment_result_size) {
        ReductionF()(output_segment_result + threadIdx.x, segment_result_slm[threadIdx.x]);
    }
}
int main(int argc, char **argv) {
    operand_type *input_data = nullptr;
    int *input_segment_id = nullptr;
    operand_type *output_segment_result = nullptr;

    operand_type *input_data_cpu = nullptr;
    int *input_segment_id_cpu = nullptr;
    operand_type *output_segment_result_cpu = nullptr;
    operand_type *output_segment_result_gold = nullptr;
    int array_num = ARRAY_SIZE;
    int segment_num = 512;
    int thread_num = 512;
    float time = 0.0f;

    cudaEvent_t event1, event2;

    std::srand(std::time(nullptr));
    int dev = cudaSetDevice(0);

    CUDA_CHECK_RETURN(cudaMalloc((void **)&input_data, array_num * sizeof(operand_type)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&input_segment_id, array_num * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&output_segment_result, segment_num * sizeof(operand_type)));

    CUDA_CHECK_RETURN(cudaEventCreate(&event1));
    CUDA_CHECK_RETURN(cudaEventCreate(&event2));

    if ((input_data_cpu = new operand_type[array_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((input_segment_id_cpu = new int[array_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((output_segment_result_cpu = new operand_type[segment_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    if ((output_segment_result_gold = new operand_type[segment_num]) == nullptr) {
        std::cout << "Allocation memory failure!" << std::endl;
        return -1;
    }
    // Initialize array
    for (int i = 0; i < array_num; i++) {
        input_data_cpu[i] = (float)rand() / RAND_MAX + 1.0f;
    }

    CUDA_CHECK_RETURN(cudaMemcpy(input_data, input_data_cpu, array_num * sizeof(operand_type), cudaMemcpyHostToDevice));

    for (int i = 0; i < array_num; i++) {
        input_segment_id_cpu[i] = rand() % segment_num;
    }
    CUDA_CHECK_RETURN(cudaMemcpy(input_segment_id, input_segment_id_cpu, array_num * sizeof(int), cudaMemcpyHostToDevice));


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    InitValue<operand_type><<<1, segment_num / PER_THREAD_INIT_NUM>>>(output_segment_result, segment_num, 1);
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Kernel InitValue Execution time " << time << " ms" << std::endl;


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    Segmentation<MulOp<operand_type>, operand_type><<<array_num / PER_THREAD_SEG_NUM / (thread_num * 2), thread_num * 2, segment_num * sizeof(operand_type)>>>(input_data, input_segment_id, output_segment_result, segment_num, array_num);
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));
    std::cout << "Kernel Segmentation Reduction time " << time << " ms" << std::endl;


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    CUDA_CHECK_RETURN(cudaMemcpy(output_segment_result_cpu, output_segment_result, segment_num * sizeof(operand_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Memory copy time: " << time << " ms" << std::endl;

    // compute and compare gold
    for (int i = 0; i < segment_num; i++) {
        output_segment_result_gold[i] = 1;
    }

    clock_t begin_clock = std::clock();
    // sum reduction
    /*
    for (int i = 0; i < array_num; i++) {
        output_segment_result_gold[input_segment_id_cpu[i]] += input_data_cpu[i];
    }
    */
    // max reduction
    /*
    for (int i = 0; i < array_num; i++) {
        output_segment_result_gold[input_segment_id_cpu[i]] = output_segment_result_gold[input_segment_id_cpu[i]] > input_data_cpu[i] ? output_segment_result_gold[input_segment_id_cpu[i]] : input_data_cpu[i];
    }
    */
    // mul reduction
    for (int i = 0; i < array_num; i++) {
        output_segment_result_gold[input_segment_id_cpu[i]] *= input_data_cpu[i];
    }

    clock_t end_clock = std::clock();
    double elapsed_cpu = double(end_clock - begin_clock) / CLOCKS_PER_SEC * 1000.0;
    std::cout << "CPU reduction time " << elapsed_cpu << " ms" << std::endl;

    for (int i = 0; i < segment_num; i++) {
        std::cout << "gpu result: " << output_segment_result_cpu[i] << " gold result: " << output_segment_result_gold[i] << std::endl;
        if ((output_segment_result_cpu[i] != output_segment_result_gold[i])) {
            std::cout << "Error comparing with Gold! " << "Position " << i << std::endl;
            return -1;
        }
    }
    std::cout << "Test passed! " << std::endl;

    // clean up
    if (input_data_cpu != nullptr) {
        delete[] input_data_cpu;
    }
    if (input_segment_id_cpu != nullptr) {
        delete[] input_segment_id_cpu;
    }
    if (output_segment_result_cpu != nullptr) {
        delete[] output_segment_result_cpu;
    }
    if (output_segment_result_gold != nullptr) {
        delete[] output_segment_result_gold;
    }

    cudaFree(output_segment_result);
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



