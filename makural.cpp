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

NeuralNetwork::NeuralNetwork(const char* input_filename,
	ActivationFunction* general_activation_function,
	ActivationFunction* output_activation_function,
	CostFunction* cost_func)

	: general_activation_func_(general_activation_function),
 	  output_activation_func_(output_activation_function),
	  cost_func_(cost_func) {

	std::ifstream inFile(input_filename);

	if (!inFile.is_open()) {
		throw "File hasn't been loaded.";
	}
	
	size_t tempNumber = 0;
	inFile >> tempNumber;
	std::vector<int> layers_sizes(tempNumber);
	for (size_t i = 0; i < layers_sizes.size(); ++i) {
		inFile >> tempNumber;
		layers_sizes[i] = tempNumber;
	}

	layers_.resize(layers_sizes.size() - 1);

	layers_[0] = new Input(layers_sizes[0], ((layers_sizes.size() > 1) ? layers_sizes[1] : 1));
	layers_[0]->loadWeightsAndBiases(inFile);

	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i] = new Dense(layers_sizes[i], layers_sizes[i + 1]);
		layers_[i]->loadWeightsAndBiases(inFile);
	}

	inFile.close();
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

	double BEGIN = -1;
	double END = 2.5;
	double dx = (END - BEGIN) / (epochs - 1);

	for (size_t j = 0; j < epochs; ++j) {
		shuffle(data_train, answers_train);

		double LEARNING_RATE = std::exp(-(BEGIN + dx * j) * (BEGIN + dx * j)) / 40.0;

		for (size_t i = 0; i < data_train.size() / butchSize; ++i) {
			optimizerSGD(data_train, i * butchSize, (i + 1) * butchSize, answers_train, LEARNING_RATE);
		}
		if (data_train.size() % butchSize != 0) {
			optimizerSGD(data_train, data_train.size() - data_train.size() % butchSize, 
				         data_train.size(), answers_train, LEARNING_RATE);
		}

		double error_test = 0.0;
		for (size_t i = 0; i < data_test.size(); ++i) {
			error_test += cost_func_->calculateCost(*answers_test[i], calculateAnswer(*data_test[i]));
		}
		error_test /= data_test.size();
		std::cout << "epoch " << j + 1 << ": error_test: " << error_test   << std::endl;
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

void NeuralNetwork::save(const char* output_filename) const {
	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) {
		throw "Can not write to file";
	}

	outFile << layers_.size() + 1 << ' ' << (layers_[0]->getAfterActivation()).columns() << ' ';
	for (int index = 0; index < layers_.size(); ++index) {
		outFile << (layers_[index]->getBeforeActivation()).columns() << ' ';
	}
	outFile << std::endl;

	for (int layer = 0; layer < layers_.size(); ++layer) {
		layers_[layer]->save(outFile) << std::endl;
	}

	outFile.close();
}

int NeuralNetwork::predict(const Matrix& input) {
	Matrix answer = calculateAnswer(input);
	int indexMax = 0;
	for (size_t i = 0; i < answer.columns(); ++i) {
		if (answer(0, indexMax) < answer(0, i)) {
			indexMax = i;
		}
	}
	return indexMax;
}

double NeuralNetwork::calculateAccuracy(const std::vector<Matrix*>& data_test, const std::vector<Matrix*> answers_test) {
	double counter = 0;
	for (size_t i = 0; i < data_test.size(); ++i) {
		if ((*answers_test[i])(0, predict((*data_test[i]))) > 0.0) {
			counter += 1;
		}
	}
	return counter / data_test.size();
}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < layers_.size(); ++i) {
		delete layers_[i];
	}
}
