/*
* Naive
* Time: 200296ms

* Naive improved
* Time: 206787ms

* Pthread
* Time: 12675ms
*/

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

void naive_MM(Matrix &left, Matrix &right, Matrix &result)
{
    int leftRows = left.getRows();
    int rightCols = right.getCols();
    int leftCols = left.getCols();

    for (int i = 0; i < leftRows; i++)
    {
        for (int j = 0; j < rightCols; j++)
        {
            for (int k = 0; k < leftCols; k++)
            {
                float new_value = result.getMatrix()[i][j] + left.getMatrix()[i][k] * right.getMatrix()[k][j];
                result.setMatrixValue(i, j, new_value);
            }
        }
    }

    for (int i = 0; i < leftRows; i++)
    {
        for (int k = 0; k < leftCols; k++)
        {
            for (int j = 0; j < rightCols; j++)
            {
                float new_value = result.getMatrix()[i][j] + left.getMatrix()[i][k] * right.getMatrix()[k][j];
                result.setMatrixValue(i, j, new_value);
            }
        }
    }
}

void improved_MM(Matrix &left, Matrix &right, Matrix &result)
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

    int leftRows = left.getRows();
    int rightCols = right.getCols();
    int leftCols = left.getCols();

    for (int i = 0; i < leftRows; i++)
    {
        for (int k = 0; k < leftCols; k++)
        {
            for (int j = 0; j < rightCols; j++)
            {
                float new_value = result.getMatrix()[i][j] + left.getMatrix()[i][k] * right.getMatrix()[k][j];
                result.setMatrixValue(i, j, new_value);
            }
        }
    }
}

typedef struct thread_arg_t
{
    int current_row;
    Matrix *left;
    Matrix *right;
    Matrix *result;
} thread_args;

void *rowCalc(void *arg)
{
    thread_args *args = (thread_args *)arg;

    int i = args->current_row;
    int res_cols = args->right->getCols();
    int mid_val = args->left->getCols();

    vector<float> temp_row = vector<float>(res_cols, 0);

    for (int k = 0; k < mid_val; k++)
    {
        float left_elem = args->left->getMatrix()[i][k];
        for (int j = 0; j < res_cols; j++)
            temp_row[j] += left_elem * args->right->getMatrix()[k][j];
    }

    for (int j = 0; j < res_cols; j++)
    {
        args->result->setMatrixValue(i, j, temp_row[j]);
    }

    return NULL;
}

void pthread_MM(Matrix &left, Matrix &right, Matrix &result)
{

    int thread_count = result.getCols();

    pthread_t threads[thread_count];
    thread_args args[thread_count];
    for (int i = 0; i < thread_count; i++)
    {
        args[i].current_row = i;
        args[i].left = &left;
        args[i].right = &right;
        args[i].result = &result;
        pthread_create(&threads[i], NULL, rowCalc, args + i);
    }
    for (int i = 0; i < thread_count; i++)
        pthread_join(threads[i], NULL);
}

int main()
{
    int N = 150;

    if (N < N_MIN || N > N_MAX)
    {
        cout << "N must be between " << N_MIN << " and " << N_MAX << endl;
        return 1;
    }

    cout << "N: " << N << endl;

    time_point start_tag;

    Matrix m1(N, N);
    m1.fill(1.0f);

    Matrix m2(N, N);
    m2.fill(1.0f);

    Matrix mr(N, N);

    MM_check(m1, m2, mr);

    // // N: 50 -> 3340ms
    // // N: 100 -> timeout
    // cout << endl
    //      << "Naive" << endl;
    // start_tag = tagTime();
    // naive_MM(m1, m2, mr);
    // tagPrint(start_tag, tagTillNow(start_tag));
    // // mr.pprintMatrix("mr");

    // // N: 50 -> 3291ms
    // // N: 100 -> timeout
    // cout << endl
    //      << "Naive improved" << endl;
    // mr.fill(0.0f);
    // start_tag = tagTime();
    // improved_MM(m1, m2, mr);
    // tagPrint(start_tag, tagTillNow(start_tag));
    // // mr.pprintMatrix("mr improved");

    // // N: 100 -> 2625ms
    // // N: 150 -> 20336ms
    // cout << endl
    //      << "Pthread" << endl;
    // mr.fill(0);
    // start_tag = tagTime();
    // pthread_MM(m1, m2, mr);
    // tagPrint(start_tag, tagTillNow(start_tag));
    // // mr.pprintMatrix("mr pthread");
}