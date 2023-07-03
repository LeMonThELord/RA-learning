#include "main.h"
#include <cstring>

#define N_MIN 10
#define N_MAX 10000

void multiply(const float *left_mat, int left_rows, int left_cols,
              const float *right_mat, int right_rows, int right_cols,
              float **result, int *r_rows, int *r_cols)
{
    if (result)
    {
        *r_rows = left_rows;
        *r_cols = right_cols;
        *result = new float[*r_rows * *r_cols];

        for (int i = 0; i < left_rows; i++)
        {
            for (int j = 0; j < right_cols; j++)
            {
                for (int k = 0; k < left_cols; k++)
                {
                    cout << i << " " << j << " " << k << endl;
                    (*result)[i * right_cols + j] += left_mat[i * left_cols + k] * right_mat[k * right_cols + j];
                }
            }
        }
    }
}

int main()
{

    int N = 3;

    float *a = new float[N * N];
    fillMatrix(a, N, N, 1.0f);

    float *b = new float[N * N];
    fillMatrix(b, N, N, 1.0f);

    float *result;
    int r_rows, r_cols;
    multiply(a, N, N, b, N, N, &result, &r_rows, &r_cols);

    pprintMatrix(a, N, N, "a");
    pprintMatrix(b, N, N, "b");
    pprintMatrix(result, N, N, "result");

    return 0;
}