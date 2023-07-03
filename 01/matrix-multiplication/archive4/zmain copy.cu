#include "matrix.cpp"
#include "tag.cpp"
#include <iostream>

using namespace std;

#define N_MIN 10
#define N_MAX 10000

void MM_check(Matrix &left, Matrix &right, Matrix &result)
{
    if (left.getCols() != right.getRows())
    {
        cout << "Error: Matrix dimensions do not match." << endl;
        exit(1);
    }

    if (right.getCols() != result.getCols() || left.getRows() != result.getRows())
    {
        cout << "Error: Result matrix has wrong dimensions." << endl;
        exit(1);
    }
}

typedef struct
{
    int rows;
    int cols;
    float *data;
} device_matrix_t;

__global__ void matrixMul(const device_matrix_t left, const device_matrix_t right, device_matrix_t result)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_val = 0;

    // Iterate over row, and down column
    for (int k = 0; k < left.cols; k++)
    {
        // Accumulate results for a single element
        temp_val += left.data[row * left.cols + k] * right.data[k * right.cols + col];

        if (row >= 0 || col >= 0)
        {
            // printf("[DEBUG]left, right = %f, %f\n", left.data[row * left.cols + k], right.data[k * right.cols + col]);
            // printf("[DEBUG]temp_val = %f\n", temp_val);
        }
    }

    // printf("[DEBUG]row, col = %d, %d\ttemp_val = %f\n", row, col, temp_val);
    result.data[row * result.cols + col] = temp_val;
    // printf("[DEBUG]result = %f\n", result.data[row * result.cols + col]);
}

void cuda_MM(Matrix &left, Matrix &right, Matrix &result)
{
    MM_check(left, right, result);

    int left_rows = left.getRows();
    int left_cols = left.getCols();
    int right_rows = right.getRows();
    int right_cols = right.getCols();
    int result_rows = result.getRows();
    int result_cols = result.getCols();
    printf("[DEBUG]cuda_MM\n");

    // Allocate memory on the device
    device_matrix_t d_left;
    d_left.rows = left_rows;
    d_left.cols = left_cols;
    int left_size = left_rows * left_cols * sizeof(float);
    cudaMalloc(&d_left.data, left_size);
    for (int i = 0; i < left_rows; i++)
    {
        for (int j = 0; j < left_cols; j++)
        {
            // cout << "[DEBUG]"
            //      << "i, j = " << i << ", " << j << "\t"
            //      << "left old: " << old_value << endl;
            cudaMemcpy(&d_left.data[i * left_cols + j], &left.getMatrix()[i][j], sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    printf("[DEBUG]d_left initialised\n");
    device_matrix_t d_right;
    d_right.rows = right_rows;
    d_right.cols = right_cols;
    int right_size = right_rows * right_cols * sizeof(float);
    cudaMalloc(&d_right.data, right_size);
    for (int i = 0; i < right_rows; i++)
    {
        for (int j = 0; j < right_cols; j++)
        {
            cudaMemcpy(&d_right.data[i * right_cols + j], &right.getMatrix()[i][j], sizeof(float), cudaMemcpyHostToDevice);
            // printf("[DEBUG]i, j = %d, %d\told_value = %f\tdevice_value = %f\n", i, j, old_value, device_value);
        }
    }
    printf("[DEBUG]d_right initialised\n");
    device_matrix_t d_result;
    d_result.rows = result_rows;
    d_result.cols = result_cols;
    int result_size = result_rows * result_cols * sizeof(float);
    cudaMalloc(&d_result.data, result_size);
    for (int i = 0; i < result_rows; i++)
    {
        for (int j = 0; j < result_cols; j++)
        {
            cudaMemset(&d_result.data[i * result_cols + j],
                       0, sizeof(float));
        }
    }
    printf("[DEBUG]d_result initialised\n");
    printf("[DEBUG]kernel start\n");
    // Launch the kernel
    int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(result_cols / dimBlock.x, result_rows / dimBlock.y);
    printf("[DEBUG]dimGrid = %d, %d\tdimBlock = %d, %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    matrixMul<<<dimGrid, dimBlock>>>(d_left, d_right, d_result);
    printf("[DEBUG]kernel ended\n");
    for (int i = 0; i < result_rows; i++)
    {
        for (int j = 0; j < result_cols; j++)
        {
            float new_value;
            cudaMemcpy(&new_value, &d_result.data[i * result_cols + j], sizeof(float), cudaMemcpyDeviceToHost);
            // cout << "[DEBUG]"
            //      << "i, j = " << i << ", " << j << "\t"
            //      << "result new: " << new_value << endl;
            result.setMatrixValue(i, j, new_value);
        }
    }

    // Free device memory
    cudaFree(d_left.data);
    cudaFree(d_right.data);
    cudaFree(d_result.data);
}

void validation(Matrix &left, Matrix &right, Matrix &result)
{

    for (int i = 0; i < left.getRows(); i++)
    {
        for (int j = 0; j < right.getCols(); j++)
        {
            float temp_value = 0;
            for (int k = 0; k < left.getCols(); k++)
            {
                temp_value += left.getMatrix()[i][k] * right.getMatrix()[k][j];
            }
            if (result.getMatrix()[i][j] != temp_value)
            {
                cout << "Error: Matrix multiplication validation failed." << endl;
                printf("[ERROR]i, j = %d, %d\n", i, j);
                printf("[ERROR]temp_value = %f\n", temp_value);
                printf("[ERROR]result = %f\n", result.getMatrix()[i][j]);
                exit(1);
            }
        }
    }
}

int main()
{
    int N = 100;

    if (N < N_MIN || N > N_MAX)
    {
        printf("N must be between %d and %d\n", N_MIN, N_MAX);
        return 1;
    }

    time_point start_tag;

    Matrix m1(N, N);
    m1.fill(1);

    Matrix m2(N, N);
    m2.fill(1);

    Matrix mr(N, N);
    mr.fill(0);

    printf("[DEBUG]starting\n");
    start_tag = tagTime();
    cuda_MM(m1, m2, mr);
    tagPrint(start_tag, tagTillNow(start_tag));
    validation(m1, m2, mr);
    // mr.pprintMatrix("mr");
}