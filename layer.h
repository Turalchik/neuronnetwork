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
	// написать в определениях функций имена аргументов
	Layer(int, int);
	virtual const Matrix& calculateOutput(const Matrix&, const ActivationFunction*) = 0;
	const Matrix& getBeforeActivation() const;
	const Matrix& getAfterActivation() const;
	const Matrix& getWeights() const;
	void putGradientIntoCurrentLayer(Matrix&& weights, Matrix&& biases);
	void setWeightsAndBiases(const Matrix& convergence_step);
};

class Input : public Layer {
public:
	Input(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateOutput(const Matrix&, const ActivationFunction*);
};

class Dense : public Layer {
public:
	Dense(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateOutput(const Matrix&, const ActivationFunction*);
};

#endif
