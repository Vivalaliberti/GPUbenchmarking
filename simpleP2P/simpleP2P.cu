/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>

// CUDA includes
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#define burst_size 64
#define interval 2
#define MAX_DATASIZE (1<<28)
#define MIN_DATASIZE (1<<8)

__global__ void SimpleKernel(int *src, int *dst, int *address)
{
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
    //dst[idx] = src[(idx%burst_size)+(address[idx/burst_size])];
}

inline bool IsAppBuiltAs64()
{
    return sizeof(void*) == 8;
}

int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);
    //printf("%d %d %d %d %d\n",sizeof(int),sizeof(int),sizeof(int),sizeof(float),sizeof(int));
    //double test = int((1<<28))*16;
   // printf("%f\n",test);
   //printf("%d\n",sizeof(int));
   //int num=int(1<<28)*4;
   //printf("%lld\n",num*sizeof(int));
   //return 1; 
    if (!IsAppBuiltAs64())
    {
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    // Number of GPUs
    printf("Checking for multiple GPUs...\n");
    int gpu_n;
    checkCudaErrors(cudaGetDeviceCount(&gpu_n));
    printf("CUDA-capable device count: %i\n", gpu_n);

    if (gpu_n < 2)
    {
        printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
        printf("Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    // Query device properties
    cudaDeviceProp prop[64];
    int gpuid[2]; // we want to find the first two GPU's that can support P2P

    for (int i=0; i < gpu_n; i++)
    {
        checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
    }
    // Check possibility for peer access
    printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

    int can_access_peer;
    int p2pCapableGPUs[2]; // We take only 1 pair of P2P capable GPUs
    p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

    // Show all the combinations of supported P2P GPUs
    for (int i = 0; i < gpu_n; i++)
    {
        for (int j = 0; j < gpu_n; j++)
        {
            if (i == j)
            {
                continue;
            }
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
            printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name, i,
                           prop[j].name, j, can_access_peer ? "Yes" : "No");
            if (can_access_peer && p2pCapableGPUs[0] == -1)
            {
                    p2pCapableGPUs[0] = i;
                    p2pCapableGPUs[1] = j;
            }
        }
    }

    if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
    {
        printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
        printf("Peer to Peer access is not available amongst GPUs in the system, waiving test.\n");

        exit(EXIT_WAIVED);
    }

    // Use first pair of p2p capable GPUs detected.
    gpuid[0] = p2pCapableGPUs[0];
    gpuid[1] = p2pCapableGPUs[1];

    // Enable peer access
    printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0], gpuid[1]);
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[1], 0));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[0], 0));
    for (int64_t num_float=MIN_DATASIZE;num_float<MAX_DATASIZE*2;num_float=int64_t(num_float*4)){
    // Allocate buffers
    const int64_t buf_size = num_float * sizeof(int);
    printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    int *g0;
    int *address;
    checkCudaErrors(cudaMalloc(&g0, buf_size*interval));
    checkCudaErrors(cudaMalloc(&address, buf_size/(burst_size*sizeof(int))*sizeof(int)));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    int *g1;
    checkCudaErrors(cudaMalloc(&g1, buf_size*interval));
    int *h0;
    int *h1;
    int *h2;
    checkCudaErrors(cudaMallocHost(&h0, buf_size*interval)); // Automatically portable with UVA
    checkCudaErrors(cudaMallocHost(&h1, buf_size/(burst_size*sizeof(int))*sizeof(int)));
    checkCudaErrors(cudaMallocHost(&h2, buf_size*interval)); // Automatically portable with UVA
    // Create CUDA event handles
    //h2 = (float *)malloc(buf_size/burst_size*sizeof(int)/sizeof(float));
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    printf("Creating event handles...\n");
    cudaEvent_t start_event, stop_event;
    float time_memcpy;
    int eventflags = cudaEventBlockingSync;
    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

    // P2P memcopy() benchmark
    //checkCudaErrors(cudaEventRecord(start_event, 0));

    //for (int i=0; i<100; i++)
    //{
        // With UVA we don't need to specify source and target devices, the
        // runtime figures this out by itself from the pointers
        // Ping-pong copy between GPUs
     //   if (i % 2 == 0)
     //   {
      //      checkCudaErrors(cudaMemcpy(g1, g0, buf_size*interval, cudaMemcpyDefault));
      //  }
     //   else
      //  {
      //      checkCudaErrors(cudaMemcpy(g0, g1, buf_size*interval, cudaMemcpyDefault));
      //  }
    //}

    //checkCudaErrors(cudaEventRecord(stop_event, 0));
    //checkCudaErrors(cudaEventSynchronize(stop_event));
    //checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    //printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
    //       (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);

    // Prepare host buffer and copy to GPU 0
    printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[1]);

    for (int i=0; i<buf_size*interval / sizeof(int); i++)
    {
        //h0[i].x = 1;
        //h0[i].y = 2;
        //h0[i].z = 3;
        //h0[i].w = 4;
        h0[i] = int(i % 4096);
    }

    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaMemcpy(g1, h0, buf_size*interval, cudaMemcpyDefault));
    for (int i=0; i<buf_size/(burst_size*sizeof(int)); i++)
    {
        h1[i] = (rand()%(burst_size*(interval-1))) + i*burst_size*interval;
    }
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaMemcpy(address, h1, buf_size/(burst_size*sizeof(int))*sizeof(int), cudaMemcpyDefault));
    // Kernel launch configuration
    const dim3 threads(512, 1);
    const dim3 blocks((buf_size / sizeof(int)) / threads.x, 1);

    // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
    // output to the GPU 1 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
           gpuid[0], gpuid[1], gpuid[0]);
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaEventRecord(start_event, 0));
    SimpleKernel<<<blocks, threads>>>(g1, g0, address);
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("kernel random access between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
           (1.0f / (time_memcpy / 1000.0f)) * (( buf_size)) / 1024.0f / 1024.0f / 1024.0f);
    // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    // output to the GPU 0 buffer
    //printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
     //      gpuid[0], gpuid[1], gpuid[0]);
    //checkCudaErrors(cudaSetDevice(gpuid[0]));
    //SimpleKernel<<<blocks, threads>>>(g1, g0);

    //checkCudaErrors(cudaDeviceSynchronize());

    // Copy data back to host and verify
    printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
    checkCudaErrors(cudaMemcpy(h2, g0, buf_size*interval, cudaMemcpyDefault));

    //int error_count = 0;

   // for (int i=0; i<buf_size / sizeof(float); i++)
    //{
        // Re-generate input data and apply 2x '* 2.0f' computation of both
        // kernel runs
      //  if (int(h2[i]) !=int(h0[(i%burst_size)+(h1[i/burst_size])]))
       // {
            //printf("Verification error @ element %d: val = %d, ref = %d, address = %d\n", i, h2[i], h0[(i%burst_size)+(h1[i/burst_size])],(i%burst_size)+(h1[i/burst_size]));

         //   if (error_count++ > 10)
           // {
         //       break;
           // }
        //}
    //}
	//if (error_count != 0)
    //{
      //  printf("Test failed!\n");
       // exit(EXIT_FAILURE);
    //}
	//else
    //{
      //  printf("Test passed\n");
       // exit(EXIT_SUCCESS);
    //}
	
	// Cleanup and shutdown
    printf("Shutting down...\n");
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaFree(g0));
    checkCudaErrors(cudaFree(address));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaFree(g1));
    checkCudaErrors(cudaFreeHost(h0));
    checkCudaErrors(cudaFreeHost(h1));
    checkCudaErrors(cudaFreeHost(h2));
    }
    // Disable peer access (also unregisters memory for non-UVA cases)
    printf("Disabling peer access...\n");
    checkCudaErrors(cudaSetDevice(gpuid[0]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[1]));
    checkCudaErrors(cudaSetDevice(gpuid[1]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[0]));



    for (int i=0; i<gpu_n; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
    }



    
}

