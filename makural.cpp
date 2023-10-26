#include "makural.h"
#include<iostream>

void NeuralNetwork::shuffle(std::vector<Matrix*>& data, std::vector<Matrix*>& answers) {
	int random = 0;
	for (int i = 0; i < data.size(); ++i) {
		random = rand() % data.size();
		if (i == random) {
			--i;
			continue;
		}
		std::swap(data[i], data[random]);
		std::swap(answers[i], answers[random]);
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers_sizes,
	ActivationFunction* general_activation_function,
	ActivationFunction* output_activation_function,
	CostFunction* cost_func)

	: layers_(layers_sizes.size() - 1),
	general_activation_func_(general_activation_function),
	output_activation_func_(output_activation_function), 
	cost_func_(cost_func) {

	layers_[0] = new Input(layers_sizes[0], ((layers_sizes.size() > 1) ? layers_sizes[1] : 1));
	layers_[0]->initializeWeightsAndBiases();

	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i] = new Dense(layers_sizes[i], layers_sizes[i + 1]);
		layers_[i]->initializeWeightsAndBiases();
	}
}

Matrix NeuralNetwork::calculateAnswer(const Matrix& input) {
	layers_[0]->calculateLayerOutput(input, general_activation_func_);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateLayerOutput(layers_[i - 1]->getBeforeActivation(), general_activation_func_);
	}
	return output_activation_func_->calculateFunction(layers_[layers_.size() - 1]->getBeforeActivation());
}

void NeuralNetwork::train(std::vector<Matrix*>& data_train, std::vector<Matrix*>& answers_train, 
						  std::vector<Matrix*>& data_test, std::vector<Matrix*>& answers_test, 
						  const size_t& epochs, const size_t& butchSize) {

	double LEARNING_RATE = 0.009;

	for (size_t j = 0; j < epochs; ++j) {
		shuffle(data_train, answers_train);
		
		/*if (j == 10) {
			LEARNING_RATE *= 10;
		}*/

		for (size_t i = 0; i < data_train.size() / butchSize; ++i) {
			optimizerSGD(data_train, i * butchSize, (i + 1) * butchSize, answers_train, LEARNING_RATE);
		}
		if (data_train.size() % butchSize != 0) {
			optimizerSGD(data_train, data_train.size() - data_train.size() % butchSize, 
				         data_train.size(), answers_train, LEARNING_RATE);
		}

		double error_test = 0.0;
		double error_train = 0.0;
		/*for (size_t i = 0; i < data_test.size(); ++i) {
			error_test += cost_func_->calculateCost(*answers_test[i], calculateAnswer(*data_test[i]));
		}*/
		for (size_t i = 0; i < data_train.size(); ++i) {
			error_train += cost_func_->calculateCost(*answers_train[i], calculateAnswer(*data_train[i]));
		}
		error_test /= data_test.size();
		error_train /= data_train.size();
		std::cout << "epoch " << j + 1 << ": error_test: " << error_test << ", error_train: " << error_train <<  std::endl;

	}

}

void NeuralNetwork::optimizerSGD(const std::vector<Matrix*>& data_train, size_t batchBegins, size_t batchEnds, 
	                             const std::vector<Matrix*>& answers, const double& learning_rate) {

	applySuitBackpropagationAlgorithm(calculateAnswer(*data_train[batchBegins]), *answers[batchBegins]);
	for (size_t i = batchBegins + 1; i < batchEnds; ++i) {
		applyGeneralBackpropagationAlgorithm(calculateAnswer(*data_train[i]), *answers[i]);
	}
	for (size_t i = 0; i < layers_.size(); ++i) {
		layers_[i]->averageGradient(batchEnds - batchBegins);
	}
	changeWeights(learning_rate);
}

void NeuralNetwork::applySuitBackpropagationAlgorithm(const Matrix& ourOutputs, const Matrix& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Matrix de_dt = ourOutputs - actualOutputs;
	Matrix de_dh = 0;
	for (size_t i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeights().multiplicationByTransposeMatrix(de_dt);

		layers_[i]->putGradientIntoCurrentLayer(layers_[i]->getAfterActivation().multiplicationTransposeByMatrix(de_dt), 
																									  std::move(de_dt));

		de_dt = de_dh.elementWiseMultiplicationTransposeByMatrix(general_activation_func_->
			calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->putGradientIntoCurrentLayer(layers_[0]->getAfterActivation().multiplicationTransposeByMatrix(de_dt), 
																								  std::move(de_dt));
}

void NeuralNetwork::applyGeneralBackpropagationAlgorithm(const Matrix& ourOutputs, const Matrix& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Matrix de_dt = ourOutputs - actualOutputs;
	Matrix de_dh = 0;
	for (size_t i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeights().multiplicationByTransposeMatrix(de_dt);

		layers_[i]->addGradientToCurrentLayer(layers_[i]->getAfterActivation().multiplicationTransposeByMatrix(de_dt), de_dt);

		de_dt = de_dh.elementWiseMultiplicationTransposeByMatrix(general_activation_func_->
			calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->addGradientToCurrentLayer(layers_[0]->getAfterActivation().multiplicationTransposeByMatrix(de_dt), de_dt);
}


void NeuralNetwork::changeWeights(const Matrix& convergence_step) {
	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i]->changeWeightsAndBiasesByGradient(convergence_step);
	}
}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < layers_.size(); ++i) {
		delete layers_[i];
	}
}
