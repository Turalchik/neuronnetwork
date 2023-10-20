#include "layer.h"

Layer::Layer(int input_size, int output_size) :
	after_activation_(1, input_size),
	before_activation_(1, output_size),
	weights_(input_size, output_size),
	biases_(1, output_size),
	gradient_nodes_weights_(input_size, output_size),
	gradient_nodes_biases_(1, output_size) {}


const Matrix& Dense::calculateOutput(const Matrix& input_data, const ActivationFunction* activationFunc) {
	if (input_data.columns() != 1 || input_data.rows() != after_activation_.rows()) {
		throw "Error in calculateOutput";
	}

	after_activation_ = activationFunc->calculateFunction(input_data);
	before_activation_ = (after_activation_ * weights_) + biases_;

	return before_activation_;
}

const Matrix& Input::calculateOutput(const Matrix& input, const ActivationFunction* activationFunc) {
	before_activation_ = (input * weights_) + biases_;
	return before_activation_;
}