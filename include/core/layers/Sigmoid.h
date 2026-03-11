#ifndef SIGMOID_H
#define SIGMOID_H

#include <core/layers/AbstractBaseLayer.h>

namespace cnncpp {

class Sigmoid : public AbstractBaseLayer {
 public:
  explicit Sigmoid(std::string name = "Sigmoid");
  ~Sigmoid() override = default;

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

#endif  // SIGMOID_H