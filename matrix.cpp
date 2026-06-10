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

  Matrix(i64 row, i64 col) : rows(row), cols(col), matrix(row * col, 0.0) {}

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
      std::cout << "\n";
    }
  }
};

int main() {
  Matrix m(2, 3);
  m.at(0, 0) = 1.0;
  m.at(1, 2) = 5.0;
  m.print_matrix();

  return 0;
}
