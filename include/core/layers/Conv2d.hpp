#ifndef CONV2D_H
#define CONV2D_H

#include "core/layers/AbstractBaseLayer.hpp"

namespace nf {

class Conv2d : public AbstractBaseLayer {
   public:
    Conv2d(size_t inChannels, size_t outChannels, size_t kernelSize = 3,
           size_t stride = 1, size_t padding = 0, bool bias = true,
           std::string name = "Conv2d");
    ~Conv2d() override = default;

    void forward(const FloatTensor& input) override;
    void backward(const FloatTensor& output_gradient) override;

    const FloatTensor& getInput() const override { return mInput; }
    const FloatTensor& getOutput() const override { return mOutput; }
    const FloatTensor& getInputGradient() const override {
        return mInputGradient;
    }

   private:
    size_t mInChannels;
    size_t mOutChannels;
    size_t mKernelSize;
    size_t mStride;
    size_t mPadding;
    bool mUseBias;

    FloatTensor mInput;
    FloatTensor mInputGradient;

    FloatTensor mKernel;
    FloatTensor mKernelGradient;

    FloatTensor mBias;
    FloatTensor mBiasGradient;

    FloatTensor mOutput;

    Shape getOutputDimsFromInput(const Shape& input);
    FloatTensor getZeroPaddedInput();
    FloatTensor convolve(const FloatTensor& input, const FloatTensor& kernel);
};

}  // namespace nf

#endif  // CONV2D_H
