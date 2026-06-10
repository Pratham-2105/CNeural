#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

using Scalar = f64;

struct Matrix {
  i64 rows, cols;
  std::vector<Scalar> matrix;

  Matrix(i64 row, i64 col, bool random)
      : rows(row), cols(col), matrix(row * col) {
    if (random) {
      std::mt19937 gen(std::random_device{}());
      std::uniform_real_distribution<Scalar> dist(-0.5, 0.5);

      for (auto &val : matrix) {
        val = dist(gen);
      }
    }
  }

  Scalar &at(i64 i, i64 j) {
    // TODO add exceptions for out-of-bounds stuff;
    return matrix[i * cols + j];
  }

  void print_matrix() {
    // TODO add exceptions for out-of-bounds stuff;
    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        std::cout << at(i, j) << " ";
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  Matrix operator+(Matrix &other_Matrix) {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("INVALID ROW or COLUMN");
    }

    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) + other_Matrix.at(i, j);
      }
    }

    return new_Matrix;
  }

  Matrix operator-(Matrix &other_Matrix) {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("INVALID ROW or COLUMN");
    }

    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) - other_Matrix.at(i, j);
      }
    }

    return new_Matrix;
  }

  Matrix scalar_multiply(f64 scalar_number) {
    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) * scalar_number;
      }
    }

    return new_Matrix;
  }
};

int main() {
  Matrix m0(2, 2, true);
  m0.at(0, 0) = 1.0;
  m0.at(0, 1) = 2.0;
  m0.at(1, 0) = 3.0;
  m0.at(1, 1) = 4.0;

  m0.print_matrix();

  Matrix m1(2, 2, true);
  m1.at(0, 0) = 1.0;
  m1.at(0, 1) = 2.0;
  m1.at(1, 0) = 3.0;
  m1.at(1, 1) = 4.0;

  // m1.print_matrix();

  /*
    Matrix addition = m0 + m1;
    addition.print_matrix();
  */

  Matrix scalar_multiplication = m0.scalar_multiply(5.0);
  scalar_multiplication.print_matrix();

  return 0;
}
