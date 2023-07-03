#ifndef MATRIX_H_
#define MATRIX_H_

#include <string>
#include <vector>
using namespace std;

class Matrix
{
protected:
    int dims[2]; // dims[0] = rows, dims[1] = cols
    vector<vector<float>> matrix;

public:
    Matrix(int dims[2]);
    Matrix(int rows, int cols);
    Matrix(vector<vector<float>> matrix);
    void randomize();
    void randomize(int seed);
    void fill(float value);
    int getRows();
    int getCols();
    vector<vector<float>> getMatrix();
    void setMatrix(vector<vector<float>> matrix);
    void printMatrix();
    void pprintMatrix(string name);
    void setMatrixValue(int row, int col, float value);
};

#endif