#include "core/layers/MaxPool.h"

#include <algorithm>
#include <limits>

namespace nf {

MaxPool::MaxPool(size_t poolSize, size_t stride, std::string name)
    : mPoolSize(poolSize), mStride(stride) {
  mName = name;
}

void MaxPool::forward(const Matrix& input) {
  mInput = input;

  // output dims
  size_t outHeight = (input.shape.x - mPoolSize) / mStride + 1;
  size_t outWidth = (input.shape.y - mPoolSize) / mStride + 1;
  mOutput.allocateMemoryIfNotAllocated({outHeight, outWidth});

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
	  size_t inpuIdx = row * input.shape.y + col;

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

void MaxPool::backward(const Matrix& output_gradient) {
  mInputGradient.allocateMemoryIfNotAllocated(mInput.shape);
  std::fill(mInputGradient.begin(), mInputGradient.end(), 0.0f);

  // Propagate gradient only to maximum elements
  for (size_t i = 0; i < mOutput.shape.x * mOutput.shape.y; i++) {
    mInputGradient[mMaxIndices[i]] += output_gradient[i];
  }
}

}  // namespace nf