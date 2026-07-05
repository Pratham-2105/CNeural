#include "matrix.hpp"
#include "mnist.hpp"
#include "network.hpp"
#include <chrono>
#include <cstddef>
#include <vector>

int main() {

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<Matrix> train_images =
      load_images("data/train-images-idx3-ubyte");
  std::vector<Matrix> train_labels =
      load_labels("data/train-labels-idx1-ubyte");

  Network net;
  net.add_layer(784, 128);
  net.add_layer(128, 10);

  Scalar lr = 0.1;
  i64 epochs = 15;

  for (i64 epoch = 0; epoch < epochs; ++epoch) {
    Scalar total_loss = 0;
    i64 correct = 0;

    for (size_t ex = 0; ex < train_images.size(); ++ex) {
      Matrix h_out = net.layers[0].forward(train_images[ex]);
      Matrix o_out = net.layers[1].forward(h_out);

      total_loss += mse_loss(o_out, train_labels[ex]);
      if (argmax(o_out) == argmax(train_labels[ex]))
        correct++;

      Matrix output_delta =
          (net.layers[1].a - train_labels[ex])
              .hadamard(net.layers[1].z.apply(sigmoid_derivative));

      Matrix hidden_delta =
          (net.layers[1].weights.transpose() * output_delta)
              .hadamard(net.layers[0].z.apply(sigmoid_derivative));
      Matrix output_bias_grad = output_delta;
      Matrix hidden_bias_grad = hidden_delta;

      Matrix output_weight_grad =
          output_delta * net.layers[1].input.transpose();
      Matrix hidden_weight_grad =
          hidden_delta * net.layers[0].input.transpose();

      net.layers[1].weights =
          net.layers[1].weights - output_weight_grad.scalar_multiply(lr);
      net.layers[1].bias =
          net.layers[1].bias - output_bias_grad.scalar_multiply(lr);

      net.layers[0].weights =
          net.layers[0].weights - hidden_weight_grad.scalar_multiply(lr);
      net.layers[0].bias =
          net.layers[0].bias - hidden_bias_grad.scalar_multiply(lr);
    }

    Scalar accuracy = (Scalar)correct / train_images.size() * 100;
    std::cout << "epoch: " << epoch << " loss: " << total_loss
              << " accuracy: " << accuracy << "%\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "training took " << seconds << " seconds\n";

  std::vector<Matrix> test_images = load_images("data/t10k-images-idx3-ubyte");
  std::vector<Matrix> test_labels = load_labels("data/t10k-labels-idx1-ubyte");

  i64 test_correct = 0;
  for (size_t ex = 0; ex < test_images.size(); ++ex) {
    Matrix h = net.layers[0].forward(test_images[ex]);
    Matrix o = net.layers[1].forward(h);

    if (argmax(o) == argmax(test_labels[ex]))
      test_correct++;
  }

  Scalar test_acc = (Scalar)test_correct / test_images.size() * 100;
  std::cout << "TEST accuracy: " << test_acc << "%\n";

  return 0;
}
