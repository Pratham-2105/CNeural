#pragma once
#include "matrix.hpp"
#include <cmath>
#include <vector>

Scalar sigmoid(Scalar x) { return 1.0 / (1.0 + std::exp(-x)); }

Scalar sigmoid_derivative(Scalar x) {
  x = sigmoid(x);
  return x * (1 - x);
}

struct Layer {
  Matrix weights, bias;
  Matrix input, z, a;

  Layer(i64 inputs, i64 outputs)
      : weights(outputs, inputs, true), bias(outputs, 1, true),
        input(1, 1, false), z(1, 1, false), a(1, 1, false) {}

  Matrix forward(const Matrix &in) {
    input = in;
    z = weights * input + bias;
    a = z.apply(sigmoid);

    return a;
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

Scalar mse_loss(const Matrix &prediction, const Matrix &target) {
  Matrix error = prediction - target;
  error = error.apply([](Scalar x) { return x * x; });

  Scalar sum = 0;

  for (Scalar it : error.matrix) {
    sum += it;
  }

  sum *= 0.5;

  return sum;
}
