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
    if (i < 0 || i >= rows) {
      throw std::invalid_argument("ROWS ARE out-of-bounds UPON at()");
    }
    if (j < 0 || j >= cols) {
      throw std::invalid_argument("COLS ARE out-of-bounds UPON at()");
    }

    return matrix[i * cols + j];
  }

  void print_matrix() {
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

  Matrix operator*(Matrix &other_Matrix) {
    if (cols != other_Matrix.rows) {
      throw std::invalid_argument(
          "INVALID DIMENSIONS FOR matrix-multiplication");
    }

    Matrix result_matrix(rows, other_Matrix.cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < other_Matrix.cols; ++j) {
        Scalar sum = 0;

        for (i64 k = 0; k < other_Matrix.rows; ++k) {
          sum += at(i, k) * other_Matrix.at(k, j);
        }

        result_matrix.at(i, j) = sum;
      }
    }

    return result_matrix;
  }

  Matrix transpose() {
    Matrix result(cols, rows, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(j, i) = at(i, j);
      }
    }

    return result;
  }

  template <typename F> Matrix apply(F func) {
    Matrix result(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(i, j) = func(at(i, j));
      }
    }

    return result;
  }
};

int main() {
  /*
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

    m1.print_matrix();
  */

  // This works for the addition of two matrices;
  /*
    Matrix addition = m0 + m1;
    addition.print_matrix();
  */

  // This is for the scalar multiplication of a matrix;
  /*
    Matrix scalar_multiplication = m0.scalar_multiply(5.0);
    scalar_multiplication.print_matrix();
  */

  Matrix a(2, 3, false);
  a.at(0, 0) = 1;
  a.at(0, 1) = 2;
  a.at(0, 2) = 3;
  a.at(1, 0) = 4;
  a.at(1, 1) = 5;
  a.at(1, 2) = 6;

  // a.print_matrix();

  Matrix b(3, 2, false);
  b.at(0, 0) = 7;
  b.at(0, 1) = 8;
  b.at(1, 0) = 9;
  b.at(1, 1) = 10;
  b.at(2, 0) = 11;
  b.at(2, 1) = 12;

  b.print_matrix();

  /*
    Matrix c = a * b;
    c.print_matrix();
  */
  /*
  Matrix d = b.transpose();
  d.print_matrix();
  */

  /*
  Matrix apply_matrix = b.apply([](Scalar x) { return x * 2; });
  apply_matrix.print_matrix();
  */
  return 0;
}
