#include "core/layers/MaxPool2d.hpp"

#include <limits>

namespace nf {

MaxPool2d::MaxPool2d(size_t poolSize, size_t stride, std::string name)
    : mPoolSize(poolSize),
      mStride(stride),
      mInput(Shape(1)),
      mOutput(Shape(1)),
      mInputGradient(Shape(1)) {
    mName = name;
}

void MaxPool2d::forward(const FloatTensor& input) {
    mInput = input;

    // output dims
    size_t outHeight = (input.shape()[0] - mPoolSize) / mStride + 1;
    size_t outWidth = (input.shape()[1] - mPoolSize) / mStride + 1;
    mOutput = FloatTensor(outHeight, outWidth);

    mMaxIndices.resize(outHeight * outWidth);

    for (size_t i = 0; i < outHeight; i++) {
        for (size_t j = 0; j < outWidth; j++) {
            float maxVal = std::numeric_limits<float>::lowest();
            size_t maxIdx = 0;

            // Find maximum in pool window
            for (size_t pi = 0; pi < mPoolSize; pi++) {
                for (size_t pj = 0; pj < mPoolSize; pj++) {
                    size_t row = i * mStride + pi;
                    size_t col = j * mStride + pj;
                    size_t inpuIdx = row * input.shape()[1] + col;

                    if (input[inpuIdx] > maxVal) {
                        maxVal = input[inpuIdx];
                        maxIdx = inpuIdx;
                    }
                }
            }

            size_t outIdx = i * outWidth + j;
            mOutput[outIdx] = maxVal;
            mMaxIndices[outIdx] = maxIdx;
        }
    }
}

void MaxPool2d::backward(const FloatTensor& output_gradient) {
    mInputGradient = FloatTensor(mInput.shape());
    mInputGradient.zero();

    // Propagate gradient only to maximum elements
    for (size_t i = 0; i < mOutput.numel(); i++) {
        mInputGradient[mMaxIndices[i]] += output_gradient[i];
    }
}

}  // namespace nf
