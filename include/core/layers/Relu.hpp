#ifndef RELU_H
#define RELU_H

#include "core/layers/AbstractBaseLayer.hpp"

namespace nf {

class Relu : public AbstractBaseLayer {
   public:
    explicit Relu(std::string name = "ReLU");
    ~Relu() override = default;

    virtual void forward(const FloatTensor& input) override;
    virtual void backward(const FloatTensor& output_gradient) override;

    virtual const FloatTensor& getInput() const override { return mInput; }
    virtual const FloatTensor& getOutput() const override { return mOutput; }
    virtual const FloatTensor& getInputGradient() const override {
        return mInputGradient;
    }

   private:
    FloatTensor mInput;
    FloatTensor mOutput;
    FloatTensor mInputGradient;
};

}  // namespace nf

#endif  // RELU_H