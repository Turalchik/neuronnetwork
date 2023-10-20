#include "makural.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers_sizes,
							 ActivationFunction* general_activation_function,
							 ActivationFunction* output_activation_function)
	
	: layers_(layers_sizes.size() - 1),
	general_activation_func_(general_activation_function),
	output_activation_func_(output_activation_function) {

	layers_[0] = new Input(layers_sizes[0], ((layers_sizes.size() > 1) ? layers_sizes[1] : 1));

	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i] = new Dense(layers_sizes[i], layers_sizes[i + 1]);
	}
}
const Matrix& NeuralNetwork::calculateOutputs(const Matrix& input) {
	layers_[0]->calculateOutput(input);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateOutput(layers_[i - 1]->getBeforeActivation(), general_activation_func_);
	}
	return layers_[layers_.size() - 1]->getBeforeActivation();
}
Matrix NeuralNetwork::calculateAnswer(const Matrix& input) {
	return output_activation_func_->calculateFunction(calculateOutputs(input));
}
void NeuralNetwork::StochasticAverageGradient(const std::vector<Matrix*>& butch, const std::vector<Matrix*>& answers, double eps, double forgetting_speed, double convergence_step) {

	double quality_functionality = 0.0;
	const CostFunction* cost_function_current_neural_network = getCostFunction();

	for (int i = 0; i < butch.size(); ++i) {
		quality_functionality += cost_function_current_neural_network->calculateCost
		(calculateAnswer(*butch[i]), *answers[i]) / static_cast<double>(butch.size());
	}

	while (quality_functionality > eps) {
		int index_random_choice_observation = rand() % butch.size();

		double cost_random_choice_observation = cost_function_current_neural_network->calculateCost
		(calculateAnswer(*butch[index_random_choice_observation]), *answers[index_random_choice_observation]);

		applyBackpropagationAlgorithm(*answers[index_random_choice_observation]);
		changeWeights(convergence_step);
		quality_functionality = forgetting_speed * cost_random_choice_observation + (1 - forgetting_speed) * quality_functionality;
	}
}
void NeuralNetwork::applyBackpropagationAlgorithm(const Matrix&) {

}
void NeuralNetwork::changeWeights(const Matrix&) {

}