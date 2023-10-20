#include "makural.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layersSizes,
							 ActivationFunction* general_activation_function,
							 ActivationFunction* output_activation_function)
	
	: layers_(layersSizes.size() - 1),
	general_activation_func_(general_activation_function),
	output_activation_func_(output_activation_function) {

	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i] = new Layer(layersSizes[i], layersSizes[i + 1]);
	}
}
const Matrix& NeuralNetwork::calculateOutputs(const Matrix& input) {
	layers_[0]->calculateOutput(input, general_activation_func_);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateOutput(layers_[i - 1]->getBeforeActivation(), general_activation_func_);
	}
	return layers_[layers_.size() - 1]->getBeforeActivation();
}
Matrix NeuralNetwork::calculateAnswer(const Matrix& input) const {
	return output_activation_func_->calculateFunction(calculateOutputs(input));
}

void NeuralNetwork::applyBackpropagationAlgorithm(const Matrix& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Matrix de_dt = output_activation_func_->calculateFunction(layers_[layers_.size() - 1]) - actualOutputs;
	Matrix de_dh = 0;
	for (int i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeighs() * transpose(de_dt);

		layers_[i]->putGradientIntoCurrentLayer(transpose(layers_[i]->getAfterActivation())) * de_dt, std::move(de_dt));

		de_dt = elementWiseMultiplication(de_dh, general_activation_func_->
			calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->putGradientIntoCurrentLayer(transpose(layers_[0]->getAfterActivation()) * de_dt, de_dt);
}


void NeuralNetwork::changeWeights(const Matrix& convergence_step) {
	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i]->setWeightsAndBiases(convergence_step);
	}
}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < layers_.size(); ++i) {
		delete layers_[i];
	}
}
