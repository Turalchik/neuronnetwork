#include "makural.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers_sizes,
	ActivationFunction* general_activation_function,
	ActivationFunction* output_activation_function,
	CostFunction* cost_func)

	: layers_(layers_sizes.size() - 1),
	general_activation_func_(general_activation_function),
	output_activation_func_(output_activation_function), 
	cost_func_(cost_func) {

	// продумать проверку во 2 аргументе
	layers_[0] = new Input(layers_sizes[0], ((layers_sizes.size() > 1) ? layers_sizes[1] : 1));

	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i] = new Dense(layers_sizes[i], layers_sizes[i + 1]);
	}
}

Matrix NeuralNetwork::calculateAnswer(const Matrix& input) {
	layers_[0]->calculateOutput(input, general_activation_func_);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateOutput(layers_[i - 1]->getBeforeActivation(), general_activation_func_);
	}
	return output_activation_func_->calculateFunction(layers_[layers_.size() - 1]->getBeforeActivation());
}

void NeuralNetwork::optimizerSGD(const std::vector<Matrix*>& butch, const std::vector<Matrix*>& answers, 
								const double& eps, const double& forgetting_speed, const double& convergence_step) {

	double quality_functionality = 0.0;
	size_t butch_size = butch.size();
	Matrix tempMatrix = 0;

	for (int i = 0; i < butch_size; ++i) {
		quality_functionality += cost_func_->calculateCost(calculateAnswer(*butch[i]), *answers[i]);
	}
	quality_functionality /= static_cast<double>(butch_size);

	while (quality_functionality > eps) {
		int index_random_choice_observation = rand() % butch_size;

		tempMatrix = calculateAnswer(*butch[index_random_choice_observation]);

		double cost_random_choice_observation = cost_func_->calculateCost(tempMatrix, *answers[index_random_choice_observation]);

		applyBackpropagationAlgorithm(tempMatrix, *answers[index_random_choice_observation]);
		changeWeights(convergence_step);

		quality_functionality = forgetting_speed * cost_random_choice_observation + (1 - forgetting_speed) * quality_functionality;
	}
}

void NeuralNetwork::applyBackpropagationAlgorithm(const Matrix& ourOutputs, const Matrix& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Matrix de_dt = ourOutputs - actualOutputs;
	Matrix de_dh = 0;
	for (int i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeights() * transpose(de_dt);

		layers_[i]->putGradientIntoCurrentLayer(transpose(layers_[i]->getAfterActivation()) * de_dt, std::move(de_dt));

		de_dt = elementWiseMultiplication(de_dh, general_activation_func_->
			calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->putGradientIntoCurrentLayer(transpose(layers_[0]->getAfterActivation()) * de_dt, std::move(de_dt));
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
