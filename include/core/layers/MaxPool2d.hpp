#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include <core/layers/AbstractBaseLayer.hpp>
#include <vector>

namespace nf {

class MaxPool2d : public AbstractBaseLayer {
   public:
    MaxPool2d(size_t poolSize = 2, size_t stride = 2,
              std::string name = "MaxPool2d");
    ~MaxPool2d() override = default;

    void forward(const Tensor& input) override;
    void backward(const Tensor& output_gradient) override;

    const Tensor& getInput() const override { return mInput; }
    const Tensor& getOutput() const override { return mOutput; }
    const Tensor& getInputGradient() const override { return mInputGradient; }

   private:
    size_t mPoolSize;
    size_t mStride;
    Tensor mInput;
    Tensor mOutput;
    Tensor mInputGradient;
    std::vector<size_t> mMaxIndices;
};

}  // namespace nf

#endif  // MAXPOOL2D_H
