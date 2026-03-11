#include "core/layers/Relu.h"

namespace cnncpp {

Relu::Relu(std::string name) { mName = name; }

void Relu::forward(const Matrix& input) {
  mInput = input;
  mOutput.allocateMemoryIfNotAllocated(input.shape);

  // mOutput = mInput > 0 ? mInput : 0
  for (size_t i = 0; i < mInput.shape.x * mInput.shape.y; i++) {
    mOutput[i] = mInput[i] > 0 ? mInput[i] : 0.0f;
  }
}

void Relu::backward(const Matrix& output_gradient) {
  mInputGradient.allocateMemoryIfNotAllocated(mInput.shape);

  // ReLU derivative: f'(x) = 1 if x > 0, else 0
  // Chain rule: dL/dx = dL/dy * dy/dx
  for (size_t i = 0; i < mInput.shape.x * mInput.shape.y; i++) {
    mInputGradient[i] = mInput[i] > 0 ? output_gradient[i] : 0.0f;
  }
}

}  // namespace cnncpp
