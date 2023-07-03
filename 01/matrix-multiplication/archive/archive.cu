#include "pthread.h"

#include <chrono>
#include <iostream>

using namespace std;

float **create_matrix(int n, int m)
{
    float **matrix = new float *[n];
    for (int i = 0; i < n; i++)
    {
        matrix[i] = new float[m];
    }
    return matrix;
}

void random_fill_matrix(float **matrix, int n, int m)
{
    std::srand(2333);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            matrix[i][j] = static_cast<float>(std::rand() % 5);
        }
    }
}

void print_matrix(float **matrix, int n, int m)
{
    cout << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << matrix[i][j] << " ";
        }

        cout << endl;
    }
}

float **single_core_MM(float **A, int An, int Am, float **B, int Bn, int Bm)
{
    // Allocate memory for the resulting matrix C
    float **C = new float *[An];
    for (int i = 0; i < An; i++)
    {
        C[i] = new float[Bm];
        for (int j = 0; j < Bm; j++)
        {
            C[i][j] = 0;
        }
    }

    // Compute the matrix multiplication
    for (int i = 0; i < An; i++)
    {
        for (int j = 0; j < Bm; j++)
        {
            for (int k = 0; k < Am; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Return the resulting matrix C
    return C;
}
struct matrix_args
{
    float **A;
    int An;
    int Am;
    float **B;
    int Bn;
    int Bm;
    float **C;
    int Cn;
    int Cm;
    int start_row;
    int end_row;
};
void *matrix_multiply(void *arg)
{
    matrix_args *args = (matrix_args *)arg;

    // Compute the elements in the given range of rows
    for (int i = args->start_row; i < args->end_row; i++)
    {
        for (int j = 0; j < args->Cm; j++)
        {
            for (int k = 0; k < args->Am; k++)
            {
                args->C[i][j] += args->A[i][k] * args->B[k][j];
            }
        }
    }

    pthread_exit(NULL);
}
float **pthread_MM(float **A, int An, int Am, float **B, int Bn, int Bm)
{

    // Allocate memory for the resulting matrix C
    float **C = create_matrix(An, Bm);
    for (int i = 0; i < An; i++)
    {
        C[i] = new float[Bm];
        for (int j = 0; j < Bm; j++)
        {
            C[i][j] = 0;
        }
    }

    // Define the number of threads and compute the chunk size
    const int num_threads = 4; // adjust this as needed
    const int chunk_size = An / num_threads;

    // Create an array of thread handles and argument structures
    pthread_t threads[num_threads];
    matrix_args args[num_threads];

    // Launch the threads in parallel
    for (int i = 0; i < num_threads; i++)
    {
        args[i].A = A;
        args[i].An = An;
        args[i].Am = Am;
        args[i].B = B;
        args[i].Bn = Bn;
        args[i].Bm = Bm;
        args[i].C = C;
        args[i].Cn = An;
        args[i].Cm = Bm;
        args[i].start_row = i * chunk_size;
        args[i].end_row = (i < num_threads - 1) ? (i + 1) * chunk_size : An;
        pthread_create(&threads[i], NULL, matrix_multiply, (void *)&args[i]);
    }

    // Wait for all the threads to finish
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Return the resulting matrix C
    return C;
}

__global__ void matrix_multiply_kernel(float *A, int An, int Am, float *B, int Bn, int Bm, float *C, int Cn, int Cm)
{
    // Compute the global row and column indices of this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the dot product of the row of A and the column of B
    if (row < Cn && col < Cm)
    {
        float sum = 0;
        for (int k = 0; k < Am; k++)
        {
            sum += A[row * Am + k] * B[k * Bm + col];
        }
        C[row * Cm + col] = sum;
    }
}

float **cuda_MM(float **A, int An, int Am, float **B, int Bn, int Bm)
{
    
    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, An * Am * sizeof(float));
    cudaMalloc((void **)&d_B, Bn * Bm * sizeof(float));
    cudaMalloc((void **)&d_C, An * Bm * sizeof(float));

    // Copy the matrices from host (CPU) memory to device (GPU)
    cudaMemcpy(d_A, A, An * Am * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bn * Bm * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block size and launch the kernel
    dim3 block_size(16, 16);
    dim3 grid_size((Bm + block_size.x - 1) / block_size.x, (An + block_size.y - 1) / block_size.y);
    matrix_multiply_kernel<<<grid_size, block_size>>>(d_A, An, Am, d_B, Bn, Bm, d_C, An, Bm);

    // Copy the result from device (GPU) memory to host (CPU)
    float *C = new float[An * Bm];
    cudaMemcpy(C, d_C, An * Bm * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory on the device (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Return the resulting matrix C
    return (float**)C;
}

int old_main()
{
    int LOW_LIMIT = 10;
    int HIGH_LIMIT = 10000;

    int N = 3;

    float **A = create_matrix(N, N);
    float **B = create_matrix(N, N);
    float **C;
    random_fill_matrix(A, N, N);
    random_fill_matrix(B, N, N);

    print_matrix(A, N, N);
    print_matrix(B, N, N);

    chrono::system_clock::time_point start;
    chrono::system_clock::time_point end;

    start = chrono::high_resolution_clock::now();
    C = single_core_MM(A, N, N, B, N, N);
    print_matrix(C, N, N);
    end = chrono::high_resolution_clock::now();
    cout << "SCMM time = " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    C = pthread_MM(A, N, N, B, N, N);
    print_matrix(C, N, N);
    end = chrono::high_resolution_clock::now();
    cout << "PTMM time = " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    C = cuda_MM(A, N, N, B, N, N);
    print_matrix(C, N, N);
    end = chrono::high_resolution_clock::now();
    cout << "CUDA time = " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    return 0;
}