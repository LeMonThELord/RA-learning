#include <iostream>

int main()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        // device name
        printf("  Device name: %s\n", prop.name);
        // compute capability
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        // clock rate
        printf("  Clock Rate (GHz): %f\n", prop.clockRate / 1.0e6);
        // number of blocks
        printf("  Max Grid Size: %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        // number of Streaming Multiprocessors
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        // number of threads per block
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        // max threads per SM
        printf("  Max Threads Per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        // shared memory per block
        printf("  Shared Memory Per Block (KB): %f\n",
               prop.sharedMemPerBlock / 1.0e3);
        // max threads per block
        printf("  Total Global Memory (GB): %f\n",
               prop.totalGlobalMem / 1.0e9);
        // max threads per block
        printf("  Total Constant Memory (KB): %f\n",
               prop.totalConstMem / 1.0e3);
        // max threads per block
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        // max threads per block
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        // max threads per block
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}
