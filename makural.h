#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"
#include"costfunctions.h"

class NeuralNetwork {
	std::vector<Layer*> layers_;
	ActivationFunction* general_activation_func_;
	ActivationFunction* output_activation_func_;
	CostFunction* cost_func_;
public:
	const CostFunction* getCostFunction() const {
		return cost_func_;
	}
	NeuralNetwork(const std::vector<int>&, ActivationFunction*, ActivationFunction*, CostFunction*);
	void StochasticAverageGradient(const std::vector<Matrix*>& butch, const std::vector<Matrix*>& answers, double eps, double forgetting_speed, double convergence_step);
	const Matrix& calculateOutputs(const Matrix&);
	Matrix calculateAnswer(const Matrix& input);
	void applyBackpropagationAlgorithm(const Matrix& actualOutputs);
	void changeWeights(const Matrix& convergence_step);
	~NeuralNetwork();
};

#endif