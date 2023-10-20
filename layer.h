#ifndef LAYER_MAKURAL
#define LAYER_MAKURAL

#include "matrixlab.h"
#include "activationFunctions.h"

class Layer {
protected:
	Matrix before_activation_;
	Matrix weights_;
	Matrix after_activation_;
	Matrix biases_;

	Matrix gradient_nodes_weights_;
	Matrix gradient_nodes_biases_;
public:
	Layer(int, int);
	virtual const Matrix& calculateOutput(const Matrix&, const ActivationFunction* = nullptr) = 0;
	const Matrix& getBeforeActivation() const {
		return before_activation_;
	}
};

class Input : public Layer {
public:
	Input(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateOutput(const Matrix&, const ActivationFunction* = nullptr);
};

class Dense : public Layer {
public:
	Dense(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateOutput(const Matrix&, const ActivationFunction* = nullptr);
};


#endif