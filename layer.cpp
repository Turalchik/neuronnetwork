#include "layer.h"
#include <random>

Layer::Layer(int input_size, int output_size) :
	after_activation_(1, input_size),
	before_activation_(1, output_size),
	weights_(input_size, output_size),
	biases_(1, output_size),
	gradient_nodes_weights_(input_size, output_size),
	gradient_nodes_biases_(1, output_size) {}

const Eigen::MatrixXd& Dense::calculateLayerOutput(const Eigen::MatrixXd& input, const ActivationFunction* activationFunc) {
	if (input.rows() != 1 || input.cols() != after_activation_.cols()) {
		throw "Error in calculateOutput";
	}

	after_activation_ = activationFunc->calculateFunction(input);
	before_activation_ = (after_activation_ * weights_) + biases_;

	return before_activation_;
}

const Eigen::MatrixXd& Input::calculateLayerOutput(const Eigen::MatrixXd& input, const ActivationFunction* activationFunc) {
	if (input.rows() != 1 || input.cols() != after_activation_.cols()) {
		throw "Error in calculateOutput";
	}
	after_activation_ = input;
	before_activation_ = (input * weights_) + biases_;

	return before_activation_;
}

const Eigen::MatrixXd& Layer::getBeforeActivation() const {
	return before_activation_;
}

const Eigen::MatrixXd& Layer::getAfterActivation() const {
	return after_activation_;
}

const Eigen::MatrixXd& Layer::getWeights() const {
	return weights_;
}

const Eigen::MatrixXd& Layer::getBiases() const {
	return biases_;
}

const Eigen::MatrixXd& Layer::getGradientsForWeights() const {
	return gradient_nodes_weights_;
}
const Eigen::MatrixXd& Layer::getGradientsForBiases() const {
	return gradient_nodes_biases_;
}

void Layer::averageGradient(const double& butch_size) {
	gradient_nodes_weights_ /= butch_size;
	gradient_nodes_biases_ /= butch_size;

}

void Layer::putGradientIntoCurrentLayer(Eigen::MatrixXd&& weights, Eigen::MatrixXd&& biases) {
	gradient_nodes_weights_ = weights;
	gradient_nodes_biases_ = biases;
}

void Layer::changeWeightsAndBiasesByGradient(const Eigen::MatrixXd& convergence_step) {
	weights_ -= convergence_step * gradient_nodes_weights_;
	biases_ -= convergence_step * gradient_nodes_biases_;
}

void Layer::addGradientToCurrentLayer(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& biases) {
	gradient_nodes_weights_ += weights;
	gradient_nodes_biases_ += biases;
}

void Layer::initializeWeightsAndBiases() {
	fillMatrixByRandomNumbers(weights_, after_activation_.cols());
	fillWithZeros(biases_);
}

void Layer::loadWeightsAndBiases(std::ifstream& inFile) {
	double tempNumber = 0;
	for (size_t i = 0; i < weights_.rows(); ++i) {
		for (size_t j = 0; j < weights_.cols(); j++) {
			inFile >> tempNumber;
			weights_(i, j) = tempNumber;
		}
	}
	for (size_t i = 0; i < biases_.rows(); ++i) {
		for (size_t j = 0; j < biases_.cols(); j++) {
			inFile >> tempNumber;
			biases_(i, j) = tempNumber;
		}
	}
}

std::ofstream& Layer::save(std::ofstream& outFile) const {
	for (int row = 0; row < weights_.rows(); ++row) {
		for (int col = 0; col < weights_.cols(); ++col) {
			outFile << weights_(row, col) << ' ';
		}
		outFile << std::endl;
	}

	outFile << std::endl;

	for (int row = 0; row < biases_.rows(); ++row) {
		for (int col = 0; col < biases_.cols(); ++col) {
			outFile << biases_(row, col) << ' ';
		}
		outFile << std::endl;
	}

	return outFile;
}

void fillMatrixByRandomNumbers(Eigen::MatrixXd& weights, const double& after_activation_size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dis(0.0, std::sqrt(2.0 / after_activation_size));

	for (size_t i = 0; i < weights.rows(); ++i) {
		for (size_t j = 0; j < weights.cols(); ++j) {
			weights(i, j) = dis(gen);
		}
	}
}

void fillWithZeros(Eigen::MatrixXd& biases) {
	for (size_t i = 0; i < biases.rows(); ++i) {
		for (size_t j = 0; j < biases.cols(); ++j) {
			biases(i, j) = 0.0;
		}
	}
}