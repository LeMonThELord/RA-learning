#include "matrix.h"
#include <iomanip>
#include <iostream>

Matrix::Matrix(int dims[2])
{
    this->dims[0] = dims[0];
    this->dims[1] = dims[1];
    this->matrix = vector<vector<float>>(dims[0], vector<float>(dims[1], 0));
}

Matrix::Matrix(int rows, int cols)
{
    this->dims[0] = rows;
    this->dims[1] = cols;
    this->matrix = vector<vector<float>>(rows, vector<float>(cols, 0));
}

Matrix::Matrix(vector<vector<float>> matrix)
{
    this->dims[0] = matrix.size();
    this->dims[1] = matrix[0].size();
    this->matrix = matrix;
}

void Matrix::randomize()
{
    for (int i = 0; i < this->dims[0]; i++)
    {
        for (int j = 0; j < this->dims[1]; j++)
        {
            this->matrix[i][j] = rand() % 20;
        }
    }
}

void Matrix::randomize(int seed)
{
    std::srand(seed);
    this->randomize();
}

int Matrix::getRows()
{
    return this->dims[0];
}

int Matrix::getCols()
{
    return this->dims[1];
}

vector<vector<float>> Matrix::getMatrix()
{
    return this->matrix;
}

void Matrix::setMatrix(vector<vector<float>> matrix)
{
    this->matrix = matrix;
}

void Matrix::printMatrix()
{

    for (int i = 0; i < this->dims[0]; i++)
    {
        for (int j = 0; j < this->dims[1]; j++)
        {
            cout << this->matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void Matrix::pprintMatrix(string name)
{

    int max_width = 0;
    for (const auto &row : this->matrix)
    {
        for (const auto &elem : row)
        {
            max_width = std::max(max_width, static_cast<int>(std::to_string(elem).size()));
        }
    }

    cout << name << endl;

    // header row

    int left_pad = std::to_string(this->dims[0]).size() + 1;

    cout << setw(left_pad) << " "
         << " |";
    for (size_t i = 0; i < this->matrix[0].size(); ++i)
    {
        cout << " " << setw(max_width) << i << "  ";
    }
    cout << endl;
    cout << string((left_pad + 2), '-')
         << string((max_width + 3) * this->dims[1], '-')
         << endl;

    // body

    for (int i = 0; i < this->dims[0]; i++)
    {
        cout << setw(left_pad) << i << " |";

        vector<float> row = this->getMatrix()[i];
        for (int j = 0; j < this->dims[1]; j++)
        {
            cout << " " << setw(max_width) << row[j] << " |";
        }
        cout << endl;
        cout << string(left_pad + 1, ' ')
             << '-'
             << string((max_width + 3) * this->dims[1], '-') << endl;
    }
}

void Matrix::setMatrixValue(int row, int col, float value)
{
    this->matrix[row][col] = value;
}
