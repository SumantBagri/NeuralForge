#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <core/layers/AbstractBaseLayer.h>

namespace nf {

class Convolution : public AbstractBaseLayer {
 public:
  Convolution(size_t inChannels, size_t outChannels, size_t kernelSize = 3,
              size_t stride = 1, size_t padding = 0, size_t filters = 1,
              bool bias = true, std::string name = "Convolution2D");
  ~Convolution() override = default;

  void forward(const Matrix& input) override;
  void backward(const Matrix& output_gradient) override;

  const Matrix& getInput() const override { return mInput; }
  const Matrix& getOutput() const override { return mOutput; }
  const Matrix& getInputGradient() const override { return mInputGradient; }

 private:
  size_t mInChannels;
  size_t mOutChannels;
  size_t mKernelSize;
  size_t mStride;
  size_t mPadding;
  size_t mFilters;
  bool mUseBias;

  Matrix mInput;
  Matrix mInputGradient;

  Matrix mKernel;
  Matrix mKernelGradient;

  Matrix mBias;
  Matrix mBiasGradient;

  Matrix mOutput;

  Shape getOutputDimsFromInput(const Shape& input);
  Matrix getZeroPaddedInput();
  Matrix convolve(const Matrix& input, const Matrix& kernel);
};

}  // namespace nf

#endif  // CONVOLUTION_H