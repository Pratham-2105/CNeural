#include "matrix.hpp"
#include "network.hpp"
#include <fstream>
#include <vector>

u32 read_u32_big_endian(std::ifstream &file) {
  unsigned char b[4];

  file.read(reinterpret_cast<char *>(b), 4);

  u32 result =
      (u32(b[0]) << 24) | (u32(b[1]) << 16) | (u32(b[2]) << 8) | u32(b[3]);
  return result;
}

Matrix one_hot(unsigned char label) {
  Matrix label_for_test(10, 1, false);
  int number = (int)label;

  label_for_test.at(number, 0) = 1.0;

  return label_for_test;
}
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
    a.at(0, 0) = 1;
    a.at(0, 1) = 2;
    a.at(1, 0) = 3;
    a.at(1, 1) = 4;
    Matrix b(2, 2, false);
    b.at(0, 0) = 1;
    b.at(0, 1) = 2;
    b.at(1, 0) = 3;
    b.at(1, 1) = 4;
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
  /*
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
  */

  // sigmoid_derivative Function Tester:

  /*
    Scalar test_sigmoid0 = 0.0;
    Scalar test_sigmoid1 = -100.0;
    Scalar test_sigmoid2 = 100.0

    std::cout << sigmoid_derivative(test_sigmoid0) << '\n';
    std::cout << sigmoid_derivative(test_sigmoid1) << '\n';
    std::cout << sigmoid_derivative(test_sigmoid2) << '\n';
  */

  // Latyers test -> single layer forward pass;
  /*
    Layer test(3, 2);
    Matrix input(3, 1, false);
    input.at(0, 0) = 1.0;
    input.at(1, 0) = 0.5;
    input.at(2, 0) = -0.5;

    Matrix output = test.forward(input);
    output.print_matrix();
  */

  // Network test -> 2 -> 3 -> 1;
  /*
    Network net;
    net.add_layer(2, 3); // 2 inputs -> 3 hidden
    net.add_layer(3, 1); // 3 hidden -> 1 output
    Matrix input(2, 1, false);
    input.at(0, 0) = 1.0;
    input.at(1, 0) = 0.0;
    Matrix output = net.forward(input);
    output.print_matrix();
  */

  /*
  Layer test(2, 3);

  Matrix in(2, 1, false);
  in.at(0, 0) = 0.5;
  in.at(1, 0) = -0.5;

  Matrix out = test.forward(in);

  std::cout << "z (pre-activation):\n";
  test.z.print_matrix();

  std::cout << "a (activation):\n";
  test.a.print_matrix();

  std::cout << "input (stored):\n";
  test.input.print_matrix();

  */

  // MSE LOSS TEST
  /*
  Matrix prediction(2, 1, false);
  prediction.at(0, 0) = 0.5;
  prediction.at(1, 0) = 0.5;

  Matrix target(2, 1, false);
  target.at(0, 0) = 1.0;
  target.at(1, 0) = 0.0;

  std::cout << "MSE loss: " << mse_loss(prediction, target) << '\n';
  */

  /*******************************************************************/

  /*
  Layer test(2, 3);
  Matrix in(2, 1, false);
  in.at(0, 0) = 0.5;
  in.at(1, 0) = -0.5;
  test.forward(in);

  Matrix y(3, 1, false);
  y.at(0, 0) = 1.0;
  y.at(1, 0) = 0.0;
  y.at(2, 0) = 1.0;

  Matrix delta = (test.a - y).hadamard(test.z.apply(sigmoid_derivative));
  delta.print_matrix();

  std::cout << "a:\n";
  test.a.print_matrix();

  std::cout << "delta:\n";
  delta.print_matrix();

  */

  /*
  Layer hidden(2, 3);
  Layer output(3, 1);

  Matrix in(2, 1, false);
  in.at(0, 0) = 0.5;
  in.at(1, 0) = -0.5;

  Matrix h_out = hidden.forward(in);
  Matrix o_out = output.forward(h_out);

  Matrix y(1, 1, false);
  y.at(0, 0) = 1.0;

  std::cout << "loss BEFORE: " << mse_loss(o_out, y) << '\n';
  // --- BP1: output layer delta ---
  Matrix output_delta =
      (output.a - y).hadamard(output.z.apply(sigmoid_derivative));

  // --- BP2: hidden layer delta ---
  Matrix hidden_delta = (output.weights.transpose() * output_delta)
                            .hadamard(hidden.z.apply(sigmoid_derivative));

  std::cout << "output_delta (1x1):\n";
  output_delta.print_matrix();
  std::cout << "hidden_delta (should be 3x1):\n";
  hidden_delta.print_matrix();

  // --- BP3: bias gradient = delta (nothing to compute) ---
  Matrix output_bias_grad = output_delta;
  Matrix hidden_bias_grad = hidden_delta;

  // --- BP4: weight gradient = delta * input^T
  Matrix output_weight_grad = output_delta * output.input.transpose();
  Matrix hidden_weight_grad = hidden_delta * hidden.input.transpose();

  // --- shape checks ---
  std::cout << "output_weight_grad (should be 1x3, matches output.weights):\n";
  output_weight_grad.print_matrix();
  std::cout << "output_bias_grad (should be 1x1):\n";
  output_bias_grad.print_matrix();

  std::cout << "hidden_weight_grad (should be 3x2, matches hidden.weights):\n";
  hidden_weight_grad.print_matrix();
  std::cout << "hidden_bias_grad (should be 3x1):\n";
  hidden_bias_grad.print_matrix();

  Scalar lr = 0.1;
  output.weights = output.weights - output_weight_grad.scalar_multiply(lr);
  output.bias = output.bias - output_bias_grad.scalar_multiply(lr);
  hidden.weights = hidden.weights - hidden_weight_grad.scalar_multiply(lr);
  hidden.bias = hidden.bias - hidden_bias_grad.scalar_multiply(lr);

  // forward AGAIN with the same input, using the updated weights
  Matrix o_out2 = output.forward(hidden.forward(in));
  std::cout << "loss AFTER:  " << mse_loss(o_out2, y) << '\n';
  */

  /*************************************************************************/
  /* FINAL XOR-TEST */

  /*
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    // XOR data
    Matrix in0(2, 1, false);
    in0.at(0, 0) = 0;
    in0.at(1, 0) = 0;
    Matrix t0(1, 1, false);
    t0.at(0, 0) = 0;
    inputs.push_back(in0);
    targets.push_back(t0);

    Matrix in1(2, 1, false);
    in1.at(0, 0) = 0;
    in1.at(1, 0) = 1;
    Matrix t1(1, 1, false);
    t1.at(0, 0) = 1;
    inputs.push_back(in1);
    targets.push_back(t1);

    Matrix in2(2, 1, false);
    in2.at(0, 0) = 1;
    in2.at(1, 0) = 0;
    Matrix t2(1, 1, false);
    t2.at(0, 0) = 1;
    inputs.push_back(in2);
    targets.push_back(t2);

    Matrix in3(2, 1, false);
    in3.at(0, 0) = 1;
    in3.at(1, 0) = 1;
    Matrix t3(1, 1, false);
    t3.at(0, 0) = 0;
    inputs.push_back(in3);
    targets.push_back(t3);

    Network net;
    net.add_layer(2, 3);
    net.add_layer(3, 1);

    Scalar lr = 0.5;
    i64 epochs = 10000;

    for (i64 epoch = 0; epoch < epochs; ++epoch) {
      Scalar total_loss = 0;

      for (size_t ex = 0; ex < inputs.size(); ++ex) {
        Matrix h_out = net.layers[0].forward(inputs[ex]);
        Matrix o_out = net.layers[1].forward(h_out);
        total_loss += mse_loss(o_out, targets[ex]);

        Matrix output_delta =
            (net.layers[1].a - targets[ex])
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

      if (epoch % 1000 == 0)
        std::cout << "epoch " << epoch << " loss " << total_loss << "\n";
    }

    std::cout << "\n--- trained predictions ---\n";
    for (size_t ex = 0; ex < inputs.size(); ++ex) {
      Matrix h = net.layers[0].forward(inputs[ex]);
      Matrix o = net.layers[1].forward(h);

      std::cout << "input " << ex << " -> " << o.at(0, 0) << " (target "
                << targets[ex].at(0, 0) << ")\n";
    }
   */

  /***********************************************************************************/
  // TRAINING ON THE MNIST DATASET;

  /*
    std::ifstream file("data/train-images-idx3-ubyte", std::ios::binary);

    if (!file) {
      std::cerr << "can't open\n";
      return 1;
    }

    u32 magic = read_u32_big_endian(file);
    u32 count = read_u32_big_endian(file);
    u32 rows = read_u32_big_endian(file);
    u32 cols = read_u32_big_endian(file);

    std::cout << magic << " " << count << " " << rows << " " << cols << "\n";

    Matrix img(784, 1, false);

    for (u32 i = 0; i < 784; ++i) {
      unsigned char pixel_byte;
      file.read(reinterpret_cast<char *>(&pixel_byte), 1);

      img.at(i, 0) = pixel_byte / 255.0;
    }

    for (i64 r = 0; r < 28; ++r) {
      for (i64 c = 0; c < 28; ++c) {
        Scalar v = img.at(r * 28 + c, 0);
        std::cout << (v > 0.5 ? '#' : ' ');
      }
      std::cout << '\n';
    }

    */

  /* READING THE LABELS ON THE FILE */

  std::ifstream labels_file("data/train-labels-idx1-ubyte", std::ios::binary);

  if (!labels_file) {
    std::cerr << "can't open labels_file\n";
    return 1;
  }

  u32 magic = read_u32_big_endian(labels_file);
  u32 count = read_u32_big_endian(labels_file);

  std::cout << magic << " " << count << "\n";

  unsigned char label;
  labels_file.read(reinterpret_cast<char *>(&label), 1);

  std::cout << (int)label << "\n";

  Matrix test_label = one_hot(label);
  test_label.print_matrix();

  return 0;
}
