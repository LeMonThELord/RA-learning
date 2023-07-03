#include <iostream>
#include <math.h>

// function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];

    __syncthreads();
}

int main(void)
{
    int N = 1 << 20; // 1M elements

    float *a;
    float *b;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float *x;
    float *y;

    cudaMalloc(&x, N * sizeof(float));
    cudaMalloc(&y, N * sizeof(float));

    cudaMemcpy(x, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    cudaMemcpy(b, y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(b[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    free(a);
    free(b);

    return 0;
}
