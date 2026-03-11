#include "core/layers/Conv2d.hpp"

#include <cmath>
#include <random>

namespace nf {

Conv2d::Conv2d(size_t inChannels, size_t outChannels, size_t kernelSize,
               size_t stride, size_t padding, size_t filters, bool bias,
               std::string name)
    : mInChannels(inChannels),
      mOutChannels(outChannels),
      mKernelSize(kernelSize),
      mStride(stride),
      mPadding(padding),
      mFilters(filters),
      mUseBias(bias),
      mInput(Shape(1)),
      mInputGradient(Shape(1)),
      mKernel(Shape(1)),
      mKernelGradient(Shape(1)),
      mBias(Shape(1)),
      mBiasGradient(Shape(1)),
      mOutput(Shape(1)) {
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
    // Kernel shape: (outChannels, inChannels * kernelSize * kernelSize)
    mKernel = Tensor(outChannels, inChannels * kernelSize * kernelSize);

    std::normal_distribution<float> kernelDist(0.0f, kernelStdDev);
    for (size_t i = 0; i < kernelElementCount; i++) {
	mKernel[i] = kernelDist(gen);
    }

    // if bias is used, allocate and initialize
    if (mUseBias) {
	float biasStdDev = std::sqrt(2.0f / fanIn) * 0.1f;
	mBias = Tensor(outChannels, 1);
	std::normal_distribution<float> biasDist(0.0f, biasStdDev);
	for (size_t i = 0; i < outChannels; i++) {
	    mBias[i] = biasDist(gen);
	}
    }
}

Shape Conv2d::getOutputDimsFromInput(const Shape& input) {
    // output_size[i] = (input_size[i] + 2*padding[i] - kernel_size[i]) /
    // stride[i] + 1
    size_t outputHeight = (input[0] + 2 * mPadding - mKernelSize) / mStride + 1;
    size_t outputWidth = (input[1] + 2 * mPadding - mKernelSize) / mStride + 1;
    return Shape(outputHeight, outputWidth);
}

Tensor Conv2d::getZeroPaddedInput() {
    Shape paddedDims = {mInput.shape()[0] + 2 * mPadding,
                        mInput.shape()[1] + 2 * mPadding};
    Tensor paddedInput(paddedDims);
    paddedInput.zero();

    // copy the input to the center of the padded input
    for (size_t i = 0; i < mInput.shape()[0]; i++) {
	for (size_t j = 0; j < mInput.shape()[1]; j++) {
	    size_t paddedIdx = (i + mPadding) * paddedDims[1] + j + mPadding;
	    size_t inputIdx = i * mInput.shape()[1] + j;
	    paddedInput[paddedIdx] = mInput[inputIdx];
	}
    }

    return paddedInput;
}

Tensor Conv2d::convolve(const Tensor& input, const Tensor& kernel) {
    Shape outputDims = getOutputDimsFromInput(input.shape());

    // initialize output tensor
    Tensor result(outputDims);

    // loop over each output cell
    for (size_t i = 0; i < outputDims[0]; i++) {
	for (size_t j = 0; j < outputDims[1]; j++) {
	    float sum = 0.0f;

	    for (size_t ki = 0; ki < mKernelSize; ki++) {
		for (size_t kj = 0; kj < mKernelSize; kj++) {
		    int row = (int)i * mStride + ki - (int)mPadding;
		    int col = (int)j * mStride + kj - (int)mPadding;

		    if (row >= 0 && row < (int)input.shape()[0] && col >= 0 &&
		        col < (int)input.shape()[1]) {
			size_t inputIdx = row * input.shape()[1] + col;
			size_t kernelIdx = ki * mKernelSize + kj;
			sum += input[inputIdx] * kernel[kernelIdx];
		    }
		}
	    }

	    result[i * outputDims[1] + j] = sum;
	}
    }

    return result;
}

void Conv2d::forward(const Tensor& input) {
    mInput = input;

    Shape outputDims = getOutputDimsFromInput(input.shape());
    // Output shape: (mOutChannels, outputHeight * outputWidth)
    mOutput = Tensor(mOutChannels, outputDims[0] * outputDims[1]);
    mOutput.zero();

    for (size_t outCh = 0; outCh < mOutChannels; outCh++) {
	for (size_t inCh = 0; inCh < mInChannels; inCh++) {
	    // extract the kernel for in<>out channel map
	    // kernel_offset = out_channel_offset + in_channel_offset
	    size_t kernelOffset =
	        outCh * mInChannels * mKernelSize * mKernelSize +
	        inCh * mKernelSize * mKernelSize;
	    // Create a view of the kernel for this channel
	    // Note: In a production framework, this would be a lightweight view
	    // without copying
	    Tensor channelKernel(mKernelSize, mKernelSize);
	    for (size_t k = 0; k < mKernelSize * mKernelSize; k++) {
		channelKernel[k] = mKernel[kernelOffset + k];
	    }

	    // convolve the input with the kernel
	    Tensor result = convolve(input, channelKernel);
	    for (size_t i = 0; i < outputDims[0] * outputDims[1]; i++) {
		mOutput[outCh * outputDims[0] * outputDims[1] + i] += result[i];
	    }
	}

	// add bias if used
	if (mUseBias) {
	    for (size_t i = 0; i < outputDims[0] * outputDims[1]; i++) {
		mOutput[outCh * outputDims[0] * outputDims[1] + i] +=
		    mBias[outCh];
	    }
	}
    }
}

void Conv2d::backward(const Tensor& output_gradient) {
    mInputGradient = Tensor(mInput.shape());
    mInputGradient.zero();

    mKernelGradient = Tensor(mKernel.shape());
    mKernelGradient.zero();

    for (size_t outCh = 0; outCh < mOutChannels; outCh++) {
	for (size_t inCh = 0; inCh < mInChannels; inCh++) {
	    size_t kernelOffset =
	        outCh * mInChannels * mKernelSize * mKernelSize +
	        inCh * mKernelSize * mKernelSize;

	    // compute input gradients
	    Tensor channelKernel(mKernelSize, mKernelSize);
	    for (size_t k = 0; k < mKernelSize * mKernelSize; k++) {
		channelKernel[k] = mKernel[kernelOffset + k];
	    }

	    Tensor channelGradient = convolve(output_gradient, channelKernel);
	    for (size_t i = 0; i < mInput.numel(); i++) {
		mInputGradient[i] += channelGradient[i];
	    }

	    Tensor inputPadded = getZeroPaddedInput();
	    Tensor inputPaddedGradient = convolve(inputPadded, output_gradient);
	}
    }
}

}  // namespace nf
