#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"

class NeuralNetwork {
	std::vector<Layer*> layers_;
	ActivationFunction* general_activation_func_;
	ActivationFunction* output_activation_func_;
public:
	NeuralNetwork(const std::vector<int>&, ActivationFunction*, ActivationFunction*);
	const Matrix& calculateOutputs(const Matrix&);
	Matrix calculateAnswer() const;

	~NeuralNetwork() {
		for (int i = 0; i < layers_.size(); ++i) {
			delete layers_[i];
		}
	}
};

#endif