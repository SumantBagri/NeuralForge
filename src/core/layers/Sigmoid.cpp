#include "core/layers/Sigmoid.h"

#include <cmath>

namespace cnncpp {

Sigmoid::Sigmoid(std::string name) { mName = name; }

void Sigmoid::forward(const Matrix& input) {
  mInput = input;
  mOutput.allocateMemoryIfNotAllocated(input.shape);

  // mOutput = 1 / 1 + e^(-input)
  for (size_t i = 0; i < mInput.shape.x * mInput.shape.y; i++) {
    mOutput[i] = 1.0f / (1.0f + std::exp(-input[i]));
  }
}

void Sigmoid::backward(const Matrix& output_gradient) {
  mInputGradient.allocateMemoryIfNotAllocated(mInput.shape);

  // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
  // Chain rule: dL/dx = dL/dy * dy/dx
  for (size_t i = 0; i < mInput.shape.x * mInput.shape.y; i++) {
    mInputGradient[i] = output_gradient[i] * mOutput[i] * (1.0f - mOutput[i]);
  }
}

}  // namespace cnncpp
