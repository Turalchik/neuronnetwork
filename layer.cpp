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

const Matrix& Layer::getBiases() const {
	return biases_;
}

const Matrix& Layer::getGradientsForWeights() const {
	return gradient_nodes_weights_;
}
const Matrix& Layer::getGradientsForBiases() const {
	return gradient_nodes_biases_;
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

void Layer::loadWeightsAndBiases(std::ifstream& inFile) {
	double tempNumber = 0;
	for (size_t i = 0; i < weights_.rows(); ++i) {
		for (size_t j = 0; j < weights_.columns(); j++) {
			inFile >> tempNumber;
			weights_(i, j) = tempNumber;
		}
	}
	for (size_t i = 0; i < biases_.rows(); ++i) {
		for (size_t j = 0; j < biases_.columns(); j++) {
			inFile >> tempNumber;
			biases_(i, j) = tempNumber;
		}
	}
}

std::ofstream& Layer::save(std::ofstream& outFile) const {
	for (int row = 0; row < weights_.rows(); ++row) {
		for (int col = 0; col < weights_.columns(); ++col) {
			outFile << weights_(row, col) << ' ';
		}
		outFile << std::endl;
	}

	outFile << std::endl;

	for (int row = 0; row < biases_.rows(); ++row) {
		for (int col = 0; col < biases_.columns(); ++col) {
			outFile << biases_(row, col) << ' ';
		}
		outFile << std::endl;
	}

	return outFile;
}