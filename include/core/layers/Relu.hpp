#ifndef RELU_H
#define RELU_H

#include <core/layers/AbstractBaseLayer.hpp>

namespace nf {

class Relu : public AbstractBaseLayer {
   public:
    explicit Relu(std::string name = "ReLU");
    ~Relu() override = default;

    virtual void forward(const Tensor& input) override;
    virtual void backward(const Tensor& output_gradient) override;

    virtual const Tensor& getInput() const override { return mInput; }
    virtual const Tensor& getOutput() const override { return mOutput; }
    virtual const Tensor& getInputGradient() const override {
	return mInputGradient;
    }

   private:
    Tensor mInput;
    Tensor mOutput;
    Tensor mInputGradient;
};

}  // namespace nf

#endif  // RELU_H