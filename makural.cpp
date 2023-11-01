#include "makural.h"
#include<iostream>

void NeuralNetwork::shuffle(std::vector<Eigen::MatrixXd*>& data, std::vector<Eigen::MatrixXd*>& answers, const size_t& from, const size_t& to) {
	int random = 0;
	for (int i = from; i < to; ++i) {
		random = rand() % to;
		if (i == random) {
			--i;
			continue;
		}
		std::swap(data[i], data[random]);
		std::swap(answers[i], answers[random]);
	}
}

ptrFun NeuralNetwork::getOptimizer(const char* optimizer_name) {
	if (!_strcmpi(optimizer_name, "stochastic") || !_strcmpi(optimizer_name, "sgd")) {
		return &NeuralNetwork::optimizerSGD;
	}
	throw "Unknown optimizer name.";
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers_sizes,
	const char* general_activation_function,
	const char* output_activation_function,
	const char* cost_function) : layers_(layers_sizes.size() - 1) {

	general_activation_func_ = HiddenActivationFunction::constructObject(general_activation_function);
	output_activation_func_ = OutputActivationFunction::constructObject(output_activation_function);
	cost_func_ = CostFunction::constructObject(cost_function);

	layers_[0] = new Input(layers_sizes[0], ((layers_sizes.size() > 1) ? layers_sizes[1] : 1));
	layers_[0]->initializeWeightsAndBiases();

	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i] = new Dense(layers_sizes[i], layers_sizes[i + 1]);
		layers_[i]->initializeWeightsAndBiases();
	}
}

NeuralNetwork::NeuralNetwork(const char* input_filename) {
	std::ifstream inFile(input_filename);
	if (!inFile.is_open()) {
		throw "File hasn't been loaded.";
	}

	std::string tempStr;
	inFile >> tempStr;
	general_activation_func_ = HiddenActivationFunction::constructObject(tempStr.c_str());
	inFile >> tempStr;
	output_activation_func_ = OutputActivationFunction::constructObject(tempStr.c_str());
	inFile >> tempStr;
	cost_func_ = CostFunction::constructObject(tempStr.c_str());

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

Eigen::MatrixXd NeuralNetwork::answerVec(const Eigen::MatrixXd& input) {
	layers_[0]->calculateLayerOutput(input, general_activation_func_);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateLayerOutput(layers_[i - 1]->getBeforeActivation(), general_activation_func_);
	}
	return output_activation_func_->calculateFunction(layers_[layers_.size() - 1]->getBeforeActivation());
}

void NeuralNetwork::train(std::vector<Eigen::MatrixXd*>& data_train, std::vector<Eigen::MatrixXd*>& answers_train, const double& validation_split,
	const char* optimizer_name, const size_t& epochs, const size_t& butchSize) {
	if (validation_split > 1 || validation_split < 0) {
		throw "Valid proportion belongs in between [0, 1].";
	}

	ptrFun optimizer = getOptimizer(optimizer_name);
	size_t real_train_size = data_train.size() * (1 - validation_split);

	srand(time(NULL));
	double LEARNING_RATE;
	double BEGIN = -1.0;
	double END = 2.5;
	double dx = epochs > 1 ? (END - BEGIN) / (epochs - 1) : 0.0;

	clock_t timeStart;
	clock_t timeFinish;

	for (size_t j = 0; j < epochs; ++j) {

		timeStart = clock();

		shuffle(data_train, answers_train, 0, real_train_size);

		LEARNING_RATE = std::exp(-(BEGIN + dx * j) * (BEGIN + dx * j)) / 20.0;
		
		for (size_t i = 0; i <  real_train_size / butchSize; ++i) {
			(this->*optimizer)(data_train, i * butchSize, (i + 1) * butchSize, answers_train, LEARNING_RATE);
		}

		if (real_train_size % butchSize != 0) {
			(this->*optimizer)(data_train, real_train_size - real_train_size % butchSize, real_train_size, answers_train, LEARNING_RATE);
		}

		timeFinish = clock();
		std::cout << "epoch=" << j + 1 << ", validation_loss=" <<
			averageLoss(data_train, answers_train, real_train_size, data_train.size()) << ", time=" <<
			static_cast<double>(timeFinish - timeStart) / CLOCKS_PER_SEC << " sec" << std::endl;


	}
}

void NeuralNetwork::optimizerSGD(const std::vector<Eigen::MatrixXd*>& data_train, size_t batchBegins, size_t batchEnds,
	                             const std::vector<Eigen::MatrixXd*>& answers, const double& learning_rate) {

	applySuitBackpropagationAlgorithm(answerVec(*data_train[batchBegins]), *answers[batchBegins]);
	for (size_t i = batchBegins + 1; i < batchEnds; ++i) {
		applyGeneralBackpropagationAlgorithm(answerVec(*data_train[i]), *answers[i]);
	}
	for (size_t i = 0; i < layers_.size(); ++i) {
		layers_[i]->averageGradient(batchEnds - batchBegins);
	}
	changeWeights(learning_rate);
}

void NeuralNetwork::applySuitBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Eigen::MatrixXd de_dt = ourOutputs - actualOutputs;
	Eigen::MatrixXd de_dh;
	for (size_t i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeights() * de_dt.transpose();

		layers_[i]->putGradientIntoCurrentLayer(layers_[i]->getAfterActivation().transpose() * de_dt, std::move(de_dt));

		de_dt = de_dh.transpose().cwiseProduct(general_activation_func_->calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->putGradientIntoCurrentLayer(layers_[0]->getAfterActivation().transpose() * de_dt, std::move(de_dt));
}

void NeuralNetwork::applyGeneralBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs) {
	//de_dt is calculated for softmax + crossentropy  /generalize
	Eigen::MatrixXd de_dt = ourOutputs - actualOutputs;
	Eigen::MatrixXd de_dh;
	for (size_t i = layers_.size() - 1; i > 0; --i) {

		de_dh = layers_[i]->getWeights() * de_dt.transpose();

		layers_[i]->addGradientToCurrentLayer(layers_[i]->getAfterActivation().transpose() * de_dt, de_dt);

		de_dt = de_dh.cwiseProduct(general_activation_func_->calculateDerivativeFunction(layers_[i - 1]->getBeforeActivation()));
	}
	layers_[0]->addGradientToCurrentLayer(layers_[0]->getAfterActivation().transpose() * de_dt, de_dt);
}


void NeuralNetwork::changeWeights(const double& convergence_step) {
	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i]->changeWeightsAndBiasesByGradient(convergence_step);
	}
}

void NeuralNetwork::save(const std::string& output_filename) const {
	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) {
		throw "Can not write to file";
	}

	outFile << general_activation_func_->getStr() << ' ' << output_activation_func_->getStr() << ' ' << cost_func_->getStr() << std::endl;

	outFile << layers_.size() + 1 << ' ' << (layers_[0]->getAfterActivation()).cols() << ' ';
	for (int index = 0; index < layers_.size(); ++index) {
		outFile << (layers_[index]->getBeforeActivation()).cols() << ' ';
	}
	outFile << std::endl;

	for (int layer = 0; layer < layers_.size(); ++layer) {
		layers_[layer]->save(outFile) << std::endl;
	}

	outFile.close();
}

int NeuralNetwork::predict(const Eigen::MatrixXd& input) {
	Eigen::MatrixXd answer = answerVec(input);
	int indexMax = 0;
	for (size_t i = 0; i < answer.cols(); ++i) {
		if (answer(0, indexMax) < answer(0, i)) {
			indexMax = i;
		}
	}
	return indexMax;
}

double NeuralNetwork::accuracy(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test) {
	double counter = 0;
	for (size_t i = 0; i < data_test.size(); ++i) {
		if ((*answers_test[i])(0, predict((*data_test[i]))) > 0.0) {
			counter += 1;
		}
	}
	return (counter / data_test.size()) * 100.0;
}

double NeuralNetwork::averageLoss(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test, 
										   const size_t& from, const size_t& to) {
	double error = 0.0;
	for (size_t i = from; i < to; ++i) {
		error += cost_func_->calculateCost(*answers_test[i], answerVec(*data_test[i]));
	}
	return error / (to - from);
}

void NeuralNetwork::reset() {
	for (size_t i = 0; i < layers_.size(); ++i) {
		layers_[i]->initializeWeightsAndBiases();
	}
}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < layers_.size(); ++i) {
		delete layers_[i];
	}
}

