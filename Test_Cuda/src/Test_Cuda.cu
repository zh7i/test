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


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define PER_THREAD_INIT_NUM 8

__global__ void InitValue(int *input, int value) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
	for (int i = 0; i < PER_THREAD_INIT_NUM; i++) {
		input[index + i * warpSize] = value;
	}
}

__global__ void Segmentation(int *input_data, int * input_index, int *output_segment_id, int segment_num) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int segment_index = input_index[index];
	int segment_value = input_data[index];
	atomicAdd(output_segment_id + segment_index, segment_value);
}

int main(int argc, char **argv) {
	int *input_data = nullptr;
	int *input_index = nullptr;
	int *output_segment_id = nullptr;

	int *input_data_cpu = nullptr;
	int array_num = 1024 * 1024 * 128;
	int segment_num = 512;
	int thread_num = 512;
    float time = 0.0f;

	cudaEvent_t event1, event2;

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = cudaSetDevice(0);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&input_data, array_num * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&input_index, array_num * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&output_segment_id, segment_num * sizeof(int)));

    CUDA_CHECK_RETURN(cudaEventCreate(&event1));
    CUDA_CHECK_RETURN(cudaEventCreate(&event2));

    CUDA_CHECK_RETURN(cudaMallocHost(&input_data_cpu, array_num * sizeof(int)));

    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    InitValue<<<array_num / thread_num, thread_num>>>(input_data, -1);
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Kernel Execution time " << time << " ms" << std::endl;


    CUDA_CHECK_RETURN(cudaEventRecord(event1, 0));
    for (int i = 0; i < 10; i++) {
    	CUDA_CHECK_RETURN(cudaMemcpy(input_data_cpu, input_data, array_num * sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_RETURN(cudaEventRecord(event2, 0));

    CUDA_CHECK_RETURN(cudaEventSynchronize(event2));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, event1, event2));

    std::cout << "Memory copy time: " << time << " ms" << std::endl;

    for (int i = 0; i < array_num; i++) {
    	if (input_data_cpu[i] != -1) {
    		std::cout << "Error comparing with Gold! " << "Position " << i << std::endl;
    		goto finish;
    	}
    }

    std::cout << "Test passed! " << std::endl;
finish:


	cudaFree(input_data_cpu);
    cudaFree(output_segment_id);
    cudaFree(input_index);
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



