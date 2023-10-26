#include "layer.h"

Layer::Layer(int input_size, int output_size) :
	after_activation_(1, input_size),
	before_activation_(1, output_size),
	weights_(input_size, output_size),
	biases_(1, output_size),
	gradient_nodes_weights_(input_size, output_size),
	gradient_nodes_biases_(1, output_size) {}

const Matrix& Dense::calculateLayerOutput(const Matrix& input, const ActivationFunction* activationFunc) {
	if (input.rows() != 1 || input.columns() != after_activation_.columns()) {
		throw "Error in calculateOutput";
	}

	after_activation_ = activationFunc->calculateFunction(input);
	before_activation_ = (after_activation_ * weights_) + biases_;

	return before_activation_;
}

const Matrix& Input::calculateLayerOutput(const Matrix& input, const ActivationFunction* activationFunc) {
	if (input.rows() != 1 || input.columns() != after_activation_.columns()) {
		throw "Error in calculateOutput";
	}
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

void Layer::averageGradient(const Matrix& butch_size) {
	gradient_nodes_weights_ /= butch_size;
	gradient_nodes_biases_ /= butch_size;

}

void Layer::putGradientIntoCurrentLayer(Matrix&& weights, Matrix&& biases) {
	gradient_nodes_weights_ = weights;
	gradient_nodes_biases_ = biases;
}

void Layer::changeWeightsAndBiasesByGradient(const Matrix& convergence_step) {
	weights_ -= convergence_step * gradient_nodes_weights_;
	biases_ -= convergence_step * gradient_nodes_biases_;
}

void Layer::addGradientToCurrentLayer(const Matrix& weights, const Matrix& biases) {
	gradient_nodes_weights_ += weights;
	gradient_nodes_biases_ += biases;
}

void Layer::initializeWeightsAndBiases() {
	weights_.FillMatrixByRandomNumbers(after_activation_.columns());
	biases_.fillWithZeros();
}