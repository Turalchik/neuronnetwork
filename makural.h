#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"

class NeuralNetwork {
	std::vector<Layer*> layers_;
	ActivationFunction* general_activation_func_;
	ActivationFunction* output_activation_func_;
	CostFunction* cost_func_;
public:
	const CostFunction* getCostFunction() const {
		return cost_func_;
	}
	NeuralNetwork(const std::vector<int>&, ActivationFunction*, ActivationFunction*);
	const Matrix& calculateOutputs(const Matrix&);
	Matrix calculateAnswer(const Matrix& input) const;
	void applyBackpropagationAlgorithm();
	void changeWeights();
	~NeuralNetwork();
};

#endif