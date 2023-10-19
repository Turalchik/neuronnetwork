#include "layer.h"

Layer::Layer(int input_size, int output_size) :
	afterActivation_(input_size, 1),
	beforeActivation_(output_size, 1),
	weights_(input_size, output_size),
	biases_(output_size, 1),
	gradientNodesWeights_(input_size, output_size),
	gradientNodesBiases_(output_size, 1) {}


const Matrix& Layer::calculateOutput(const Matrix& input_data) {
	if (input_data.columns() != 1 || input_data.rows() != afterActivation_.rows()) {
		throw "Error in calculateOutput";
	}

	afterActivation_ = calculateFunction(input_data);
	for (int output = 0; output < beforeActivation_.rows(); ++output) {
		for (int input = 0; input < afterActivation_.rows(); ++input) {
			beforeActivation_(output, 0) += weights_(input, output) * afterActivation_(input, 0);
		}
	}

	return beforeActivation_;
}