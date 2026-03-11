#include "core/layers/Sigmoid.hpp"

#include <cmath>

namespace nf {

Sigmoid::Sigmoid(std::string name)
    : mInput(Shape(1)), mOutput(Shape(1)), mInputGradient(Shape(1)) {
    mName = name;
}

void Sigmoid::forward(const FloatTensor& input) {
    mInput = input;
    // Output has same shape as input
    mOutput = FloatTensor(input.shape(), input.device());

    // mOutput = 1 / 1 + e^(-input)
    for (size_t i = 0; i < mInput.numel(); i++) {
        mOutput[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void Sigmoid::backward(const FloatTensor& output_gradient) {
    mInputGradient = FloatTensor(mInput.shape(), mInput.device());

    // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    // Chain rule: dL/dx = dL/dy * dy/dx
    for (size_t i = 0; i < mInput.numel(); i++) {
        mInputGradient[i] =
            output_gradient[i] * mOutput[i] * (1.0f - mOutput[i]);
    }
}

}  // namespace nf
