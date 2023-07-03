#include "matrix.cpp"
#include "tag.cpp"
#include <iostream>

using namespace std;

#define N_MIN 1
#define N_MAX 1000

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
                result.setMatrixValue(i, j, left.getMatrix()[i][k] * right.getMatrix()[k][j]);
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
                result.setMatrixValue(i, j, left.getMatrix()[i][k] * right.getMatrix()[k][j]);
            }
        }
    }
}

void pthread_MM(Matrix &left, Matrix &right, Matrix &result)
{
}

int main()
{
    int N = 100;

    if (N < N_MIN || N > N_MAX)
    {
        cout << "N must be between " << N_MIN << " and " << N_MAX << endl;
        return 1;
    }

    time_point start_tag;

    Matrix m1(N, N);
    m1.randomize(2333);

    Matrix m2(N, N);
    m2.randomize(23333);

    Matrix mr(N, N);

    MM_check(m1, m2, mr);

    start_tag = tagTime();
    naive_MM(m1, m2, mr);
    tagPrint(start_tag, tagTillNow(start_tag));
    // mr.pprintMatrix("mr");

    start_tag = tagTime();
    improved_MM(m1, m2, mr);
    tagPrint(start_tag, tagTillNow(start_tag));
    // mr.pprintMatrix("mr improved");
}