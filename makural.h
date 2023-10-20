#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"
#include "costfunctions.h"

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
	Matrix calculateAnswer(const Matrix&);
	void applyBackpropagationAlgorithm(const Matrix&);
	void changeWeights(const Matrix&);
	void StochasticAverageGradient(const std::vector<Matrix*>& butch, const std::vector<Matrix*>& answers, double eps, double forgetting_speed, double convergence_step);

	~NeuralNetwork() {
		for (int i = 0; i < layers_.size(); ++i) {
			delete layers_[i];
		}
	}
};



#endif