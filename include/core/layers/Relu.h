#ifndef RELU_H
#define RELU_H

#include <core/layers/AbstractBaseLayer.h>

namespace cnncpp {

class Relu : public AbstractBaseLayer {
 public:
  explicit Relu(std::string name = "ReLU");
  ~Relu() override = default;

  virtual void forward(const Matrix& input) override;
  virtual void backward(const Matrix& output_gradient) override;

  virtual const Matrix& getInput() const override { return mInput; }
  virtual const Matrix& getOutput() const override { return mOutput; }
  virtual const Matrix& getInputGradient() const override {
    return mInputGradient;
  }

 private:
  Matrix mInput;
  Matrix mOutput;
  Matrix mInputGradient;
};

}  // namespace cnncpp

#endif  // RELU_H