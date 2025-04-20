#include <cuda_runtime.h> 
#include <nccl.h> 

// CUDA kernel for scaling data 
global void scaleKernel(float* data, int size, float scale) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < size) { 
        data[idx] *= scale; 
    } 
}

// Wrapper for NCCL all-gather (for inter-node communication) 
extern "C" { 
    void allGatherWrapper(float* sendbuf, float* recvbuf, int size, ncclComm_t comm, cudaStream_t stream) { 
        ncclResult_t ncclErr = ncclAllGather(sendbuf, recvbuf, size, ncclFloat, comm, stream); 
        if (ncclErr != ncclSuccess) { 
            fprintf(stderr, "NCCL AllGather Error: %d\n", ncclErr); exit(-1); 
        } 
    } 
}