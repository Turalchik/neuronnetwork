#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"

class NeuralNetwork {
	std::vector<Layer*> layers_;
public:
	NeuralNetwork(const std::vector<int>&);
	const Matrix& calculateOutputs(const Matrix&);
};

#endif