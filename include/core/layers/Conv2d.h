#ifndef CONV2D_H
#define CONV2D_H

#include <core/layers/AbstractBaseLayer.h>

namespace nf {

class Conv2d : public AbstractBaseLayer {
 public:
  Conv2d(size_t inChannels, size_t outChannels, size_t kernelSize = 3,
         size_t stride = 1, size_t padding = 0, size_t filters = 1,
         bool bias = true, std::string name = "Conv2d");
  ~Conv2d() override = default;

  void forward(const Tensor& input) override;
  void backward(const Tensor& output_gradient) override;

  const Tensor& getInput() const override { return mInput; }
  const Tensor& getOutput() const override { return mOutput; }
  const Tensor& getInputGradient() const override { return mInputGradient; }

 private:
  size_t mInChannels;
  size_t mOutChannels;
  size_t mKernelSize;
  size_t mStride;
  size_t mPadding;
  size_t mFilters;
  bool mUseBias;

  Tensor mInput;
  Tensor mInputGradient;

  Tensor mKernel;
  Tensor mKernelGradient;

  Tensor mBias;
  Tensor mBiasGradient;

  Tensor mOutput;

  Shape getOutputDimsFromInput(const Shape& input);
  Tensor getZeroPaddedInput();
  Tensor convolve(const Tensor& input, const Tensor& kernel);
};

}  // namespace nf

#endif  // CONV2D_H
