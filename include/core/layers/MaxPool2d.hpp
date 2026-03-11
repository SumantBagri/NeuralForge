#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include <vector>

#include "core/layers/AbstractBaseLayer.hpp"

namespace nf {

class MaxPool2d : public AbstractBaseLayer {
   public:
    MaxPool2d(size_t poolSize = 2, size_t stride = 2,
              std::string name = "MaxPool2d");
    ~MaxPool2d() override = default;

    void forward(const FloatTensor& input) override;
    void backward(const FloatTensor& output_gradient) override;

    const FloatTensor& getInput() const override { return mInput; }
    const FloatTensor& getOutput() const override { return mOutput; }
    const FloatTensor& getInputGradient() const override {
        return mInputGradient;
    }

   private:
    size_t mPoolSize;
    size_t mStride;
    FloatTensor mInput;
    FloatTensor mOutput;
    FloatTensor mInputGradient;
    std::vector<size_t> mMaxIndices;
};

}  // namespace nf

#endif  // MAXPOOL2D_H
