#pragma once
#include "matrix.hpp"
#include <cmath>
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

Scalar sigmoid(Scalar x) { return 1.0 / (1.0 + std::exp(-x)); }

Scalar sigmoid_derivative(Scalar x) {
  x = sigmoid(x);
  return x * (1 - x);
}

struct Layer {
  Matrix weights, bias;

  Layer(i64 inputs, i64 outputs)
      : weights(outputs, inputs, true), bias(outputs, 1, true) {}

  Matrix forward(Matrix &input) {
    return (weights * input + bias).apply(sigmoid);
  }
};

struct Network {
  std::vector<Layer> layers;

  Network() {}

  void add_layer(i64 inputs, i64 outputs) {
    layers.push_back(Layer(inputs, outputs));
  }

  Matrix forward(Matrix &input) {
    Matrix current = input;
    for (auto &layer : layers) {
      current = layer.forward(current);
    }

    return current;
  }
};
