#include "layer.h"

Layer::Layer(int input_size, int output_size) :
	after_activation_(1, input_size),
	before_activation_(1, output_size),
	weights_(input_size, output_size),
	biases_(1, output_size),
	gradient_nodes_weights_(input_size, output_size),
	gradient_nodes_biases_(1, output_size) {}

// можно переназвать типо fillLayer или что-то такое, чтоб понятнее было
const Matrix& Dense::calculateOutput(const Matrix& input_data, const ActivationFunction* activationFunc) {
	if (input_data.columns() != 1 || input_data.rows() != after_activation_.rows()) {
		throw "Error in calculateOutput";
	}

	after_activation_ = activationFunc->calculateFunction(input_data);
	before_activation_ = (after_activation_ * weights_) + biases_;

	return before_activation_;
}

// здесь тоже проверку на ошибки
const Matrix& Input::calculateOutput(const Matrix& input, const ActivationFunction* activationFunc) {
	after_activation_ = input;
	before_activation_ = (input * weights_) + biases_;

	return before_activation_;
}

const Matrix& Layer::getBeforeActivation() const {
	return before_activation_;
}

const Matrix& Layer::getAfterActivation() const {
	return after_activation_;
}

const Matrix& Layer::getWeights() const {
	return weights_;
}

void Layer::putGradientIntoCurrentLayer(Matrix&& weights, Matrix&& biases) {
	gradient_nodes_weights_ = weights;
	gradient_nodes_biases_ = biases;
}

void Layer::setWeightsAndBiases(const Matrix& convergence_step) {
	weights_ -= convergence_step * gradient_nodes_weights_;
	biases_ -= convergence_step * gradient_nodes_biases_;
}