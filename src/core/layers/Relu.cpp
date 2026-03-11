#include "core/layers/Relu.hpp"

namespace nf {

Relu::Relu(std::string name)
    : mInput(Shape(1)), mOutput(Shape(1)), mInputGradient(Shape(1)) {
    mName = name;
}

void Relu::forward(const Tensor& input) {
    mInput = input;
    // Output has same shape as input
    mOutput = Tensor(input.shape(), input.dtype(), input.device());

    // mOutput = mInput > 0 ? mInput : 0
    for (size_t i = 0; i < mInput.numel(); i++) {
	mOutput[i] = mInput[i] > 0 ? mInput[i] : 0.0f;
    }
}

void Relu::backward(const Tensor& output_gradient) {
    mInputGradient = Tensor(mInput.shape(), mInput.dtype(), mInput.device());

    // ReLU derivative: f'(x) = 1 if x > 0, else 0
    // Chain rule: dL/dx = dL/dy * dy/dx
    for (size_t i = 0; i < mInput.numel(); i++) {
	mInputGradient[i] = mInput[i] > 0 ? output_gradient[i] : 0.0f;
    }
}

}  // namespace nf
