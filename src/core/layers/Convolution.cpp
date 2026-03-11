#include "core/layers/Convolution.h"

#include <cmath>
#include <random>

namespace nf {

Convolution::Convolution(size_t inChannels, size_t outChannels,
                         size_t kernelSize, size_t stride, size_t padding,
                         size_t filters, bool bias, std::string name)
    : mInChannels(inChannels),
      mOutChannels(outChannels),
      mKernelSize(kernelSize),
      mStride(stride),
      mPadding(padding),
      mFilters(filters),
      mUseBias(bias) {
  mName = name;

  // He initialization parameters
  // we only have the ReLU activation for now
  // and He initialization is better compared to Xavier
  float fanIn = inChannels * kernelSize * kernelSize;
  float kernelStdDev = std::sqrt(2.0f / fanIn);

  std::random_device rd;
  std::mt19937 gen(rd());

  // allocate kernel memory
  size_t kernelElementCount =
      outChannels * inChannels * kernelSize * kernelSize;
  mKernel.allocateMemoryIfNotAllocated(
      {outChannels, inChannels * kernelSize * kernelSize});

  std::normal_distribution<float> kernelDist(0.0f, kernelStdDev);
  for (size_t i = 0; i < kernelElementCount; i++) {
    mKernel[i] = kernelDist(gen);
  }

  // if bias is used, allocate and initialize
  if (mUseBias) {
    float biasStdDev = std::sqrt(2.0f / fanIn) * 0.1f;
    mBias.allocateMemoryIfNotAllocated({outChannels, 1});
    std::normal_distribution<float> biasDist(0.0f, biasStdDev);
    for (size_t i = 0; i < outChannels; i++) {
      mBias[i] = biasDist(gen);
    }
  }
}

Shape Convolution::getOutputDimsFromInput(const Shape& input) {
  // output_size[i] = (input_size[i] + 2*padding[i] - kernel_size[i]) /
  // stride[i] + 1
  size_t outputHeight = (input.x + 2 * mPadding - mKernelSize) / mStride + 1;
  size_t outputWidth = (input.y + 2 * mPadding - mKernelSize) / mStride + 1;
  return Shape(outputHeight, outputWidth);
}

Matrix Convolution::getZeroPaddedInput() {
  Shape paddedDims = {mInput.shape.x + 2 * mPadding,
                      mInput.shape.y + 2 * mPadding};
  Matrix paddedInput(paddedDims);
  paddedInput.allocateMemory();
  std::fill(paddedInput.begin(), paddedInput.end(), 0.0f);

  // copy the input to the center of the padded input
  for (size_t i = 0; i < mInput.shape.x; i++) {
    for (size_t j = 0; j < mInput.shape.y; j++) {
      size_t paddedIdx = (i + mPadding) * paddedDims.y + j + mPadding;
      size_t inputIdx = i * mInput.shape.y + j;
      paddedInput[paddedIdx] = mInput[inputIdx];
    }
  }

  return paddedInput;
}

Matrix Convolution::convolve(const Matrix& input, const Matrix& kernel) {
  Shape outputDims = getOutputDimsFromInput(input.shape);

  // initialize output matrix
  Matrix result(outputDims);
  result.allocateMemory();

  // loop over each output cell
  for (size_t i = 0; i < outputDims.x; i++) {
    for (size_t j = 0; j < outputDims.y; j++) {
      float sum = 0.0f;

      for (size_t ki = 0; ki < mKernelSize; ki++) {
	for (size_t kj = 0; kj < mKernelSize; kj++) {
	  int row = (int)i * mStride + ki - (int)mPadding;
	  int col = (int)j * mStride + kj - (int)mPadding;

	  if (row >= 0 && row < (int)input.shape.x && col >= 0 &&
	      col < (int)input.shape.y) {
	    size_t inputIdx = row * input.shape.y + col;
	    size_t kernelIdx = ki * mKernelSize + kj;
	    sum += input[inputIdx] * kernel[kernelIdx];
	  }
	}
      }

      result[i * outputDims.y + j] = sum;
    }
  }

  return result;
}

void Convolution::forward(const Matrix& input) {
  mInput = input;

  Shape outputDims = getOutputDimsFromInput(input.shape);
  mOutput.allocateMemoryIfNotAllocated(
      {mOutChannels, outputDims.x * outputDims.y});
  std::fill(mOutput.begin(), mOutput.end(), 0.0f);

  for (size_t outCh = 0; outCh < mOutChannels; outCh++) {
    for (size_t inCh = 0; inCh < mInChannels; inCh++) {
      // extract the kernel for in<>out channel map
      // kernel_offset = out_channel_offset + in_channel_offset
      size_t kernelOffset = outCh * mInChannels * mKernelSize * mKernelSize +
                            inCh * mKernelSize * mKernelSize;
      Matrix channelKernel(mKernelSize, mKernelSize);

      for (size_t k = 0; k < mKernelSize * mKernelSize; k++) {
	channelKernel[k] = mKernel[kernelOffset + k];
      }

      // convolve the input with the kernel
      Matrix result = convolve(input, channelKernel);
      for (size_t i = 0; i < outputDims.x * outputDims.y; i++) {
	mOutput[outCh * outputDims.x * outputDims.y + i] += result[i];
      }
    }

    // add bias if used
    if (mUseBias) {
      for (size_t i = 0; i < outputDims.x * outputDims.y; i++) {
	mOutput[outCh * outputDims.x * outputDims.y + i] += mBias[outCh];
      }
    }
  }
}

void Convolution::backward(const Matrix& output_gradient) {
  mInputGradient.allocateMemoryIfNotAllocated(mInput.shape);
  std::fill(mInputGradient.begin(), mInputGradient.end(), 0.0f);

  mKernelGradient.allocateMemoryIfNotAllocated(mKernel.shape);
  std::fill(mKernelGradient.begin(), mKernelGradient.end(), 0.0f);

  for (size_t outCh = 0; outCh < mOutChannels; outCh++) {
    for (size_t inCh = 0; inCh < mInChannels; inCh++) {
      size_t kernelOffset = outCh * mInChannels * mKernelSize * mKernelSize +
                            inCh * mKernelSize * mKernelSize;

      // compute input gradients
      Matrix channelKernel(mKernelSize, mKernelSize);
      for (size_t k = 0; k < mKernelSize * mKernelSize; k++) {
	channelKernel[k] = mKernel[kernelOffset + k];
      }

      Matrix channelGradient = convolve(output_gradient, channelKernel);
      for (size_t i = 0; i < mInput.shape.x * mInput.shape.y; i++) {
	mInputGradient[i] += channelGradient[i];
      }

      Matrix inputPadded = getZeroPaddedInput();
      Matrix inputPaddedGradient = convolve(inputPadded, output_gradient);
    }
  }
}

}  // namespace nf
