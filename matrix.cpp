// ============================================================================
//  CNeural — matrix.cpp
//  Stage 0: a from-scratch matrix math library in C++. No libraries.
//  Every operation returns a NEW matrix (non-destructive); originals untouched.
//  Storage is flat row-major (matrix[i * cols + j]), accessed via at(i, j).
// ============================================================================

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ---- fixed-width type aliases ----------------------------------------------
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

using Scalar = f64; // one knob to switch the whole library's precision

// ============================================================================
//  Matrix
// ============================================================================

struct Matrix {
  i64 rows, cols;
  std::vector<Scalar> matrix; // flat, row-major

  // Construct rows x cols. random=true fills with uniform values in [-0.5,
  // 0.5]; random=false leaves it zero-initialised.
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

  // Read/write a single element. Bounds-checked.
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

  // ---- element-wise addition (same dimensions) -----------------------------
  // Used in the forward pass to add the bias: W*input + b.
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

  // ---- element-wise subtraction (same dimensions) --------------------------
  // Used for error (prediction - target) and the weight update.
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

  // ---- scalar multiply (every element x a number) --------------------------
  // Used for the gradient-descent step: weights - learning_rate * gradient.
  Matrix scalar_multiply(f64 scalar_number) {
    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) * scalar_number;
      }
    }

    return new_Matrix;
  }

  // ---- matrix multiplication: (m x n)(n x p) = (m x p) ---------------------
  // THE forward pass. Inner dimensions must match (cols == other.rows).
  // NOTE: this is the dot-product product, NOT element-wise (see hadamard).
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

  // ---- transpose: (m x n) -> (n x p) ---------------------------------------
  // Needed in backprop to push error backward through a layer.
  Matrix transpose() {
    Matrix result(cols, rows, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(j, i) = at(i, j);
      }
    }

    return result;
  }

  // ---- apply: run a Scalar->Scalar function on every element ---------------
  // How activations (sigmoid, ReLU, ...) hit a whole matrix. Templated so the
  // function is resolved at compile time and inlined.
  template <typename F> Matrix apply(F func) {
    Matrix result(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(i, j) = func(at(i, j));
      }
    }

    return result;
  }

  // ---- hadamard: element-wise multiply (same dimensions) -------------------
  // NOT matrix multiply. Used in backprop: error (x) activation-derivative.
  Matrix hadamard(Matrix &other_Matrix) {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("ROWS or COLS must be same for hadamard");
    }

    Matrix result(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(i, j) = at(i, j) * other_Matrix.at(i, j);
      }
    }

    return result;
  }
};

// ============================================================================
//  main — usage examples + checks (each verified by hand on known values)
//  Uncomment a block to run that example.
// ============================================================================
int main() {
  // ---- addition
  // --------------------------------------------------------------
  // [[1,2],[3,4]] + [[1,2],[3,4]] = [[2,4],[6,8]]
  /*
    Matrix a(2, 2, false);
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Matrix b(2, 2, false);
    b.at(0,0)=1; b.at(0,1)=2; b.at(1,0)=3; b.at(1,1)=4;
    Matrix sum = a + b;
    sum.print_matrix();
  */

  // ---- subtraction
  // -----------------------------------------------------------
  // [[5,5],[5,5]] - [[1,2],[3,4]] = [[4,3],[2,1]]
  /*
    Matrix a(2, 2, false);
    a.at(0,0)=5; a.at(0,1)=5; a.at(1,0)=5; a.at(1,1)=5;
    Matrix b(2, 2, false);
    b.at(0,0)=1; b.at(0,1)=2; b.at(1,0)=3; b.at(1,1)=4;
    Matrix diff = a - b;
    diff.print_matrix();
  */

  // ---- scalar multiply
  // -------------------------------------------------------
  // [[1,2],[3,4]] * 5 = [[5,10],[15,20]]
  /*
    Matrix a(2, 2, false);
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Matrix scaled = a.scalar_multiply(5.0);
    scaled.print_matrix();
  */

  // ---- matrix multiply
  // ------------------------------------------------------- (2x3)(3x2):
  // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
  /*
    Matrix a(2, 3, false);
    a.at(0,0)=1; a.at(0,1)=2; a.at(0,2)=3;
    a.at(1,0)=4; a.at(1,1)=5; a.at(1,2)=6;
    Matrix b(3, 2, false);
    b.at(0,0)=7;  b.at(0,1)=8;
    b.at(1,0)=9;  b.at(1,1)=10;
    b.at(2,0)=11; b.at(2,1)=12;
    Matrix product = a * b;
    product.print_matrix();
  */

  // ---- transpose
  // -------------------------------------------------------------
  // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
  /*
    Matrix a(2, 3, false);
    a.at(0,0)=1; a.at(0,1)=2; a.at(0,2)=3;
    a.at(1,0)=4; a.at(1,1)=5; a.at(1,2)=6;
    Matrix t = a.transpose();
    t.print_matrix();
  */

  // ---- apply (with a lambda)
  // ------------------------------------------------- double each element:
  // [[1,2],[3,4]] -> [[2,4],[6,8]]
  /*
    Matrix a(2, 2, false);
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Matrix doubled = a.apply([](Scalar x){ return x * 2; });
    doubled.print_matrix();
  */

  // ---- hadamard (element-wise multiply)
  // --------------------------------------
  // [[1,2],[3,4]] (.) [[5,6],[7,8]] = [[5,12],[21,32]]
  Matrix a(2, 2, false);
  a.at(0, 0) = 1;
  a.at(0, 1) = 2;
  a.at(1, 0) = 3;
  a.at(1, 1) = 4;
  Matrix b(2, 2, false);
  b.at(0, 0) = 5;
  b.at(0, 1) = 6;
  b.at(1, 0) = 7;
  b.at(1, 1) = 8;
  Matrix h = a.hadamard(b);
  h.print_matrix();

  return 0;
}
