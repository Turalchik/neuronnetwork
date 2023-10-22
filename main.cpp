#include"makural.h"
#include<fstream>
#include<iostream>

enum CONSTANTS
{
	TRAIN_DATA_VOLUME = 10'000,
	INPUT_NEURON_AMOUNT = 784,
	OUTPUT_NEURON_AMOUNT = 10,
	BUTCH_SIZE = 100,
	EPOCHS_AMOUNT = 20,
};

void shuffle(std::vector<Matrix*>& data, std::vector<Matrix*>& answers) {
	int random = 0;
	for (int i = 0; i < TRAIN_DATA_VOLUME; ++i) {
		random = rand() % TRAIN_DATA_VOLUME;
		if (i == random) {
			--i;
			continue;
		}
		std::swap(data[i], data[random]);
		std::swap(answers[i], answers[random]);
	}
}

int main() {
	srand(time(NULL));
	std::ifstream train_dataset("mnist_test.csv");
	std::ifstream test_dataset("mnist_test.csv");
	std::vector<Matrix*> data(TRAIN_DATA_VOLUME);
	std::vector<Matrix*> answers(TRAIN_DATA_VOLUME);

	NeuralNetwork makural({784, 128, 32, 10}, new ReLu, new Softmax, new CrossEntropy);

	size_t rowIndex = 0;
	int get = 0;

	while (train_dataset.get() != '\n') {}

	while (!train_dataset.eof()) {
		
		answers[rowIndex] = new Matrix(1, OUTPUT_NEURON_AMOUNT);
		(*answers[rowIndex])(0, train_dataset.get() - '0') = 1.0;
		train_dataset.ignore();

		data[rowIndex] = new Matrix(1, INPUT_NEURON_AMOUNT);
		for (int i = 0; i < INPUT_NEURON_AMOUNT; ++i) {
			train_dataset >> get;
			(*data[rowIndex])(0, i) = static_cast<double>(get) / 255.0;
			train_dataset.ignore();
		}
		++rowIndex;
		while ((isdigit(train_dataset.peek()) == 0) && train_dataset.get() != EOF) {}
	}

	for (int j = 0; j < EPOCHS_AMOUNT; ++j) {
		shuffle(data, answers);
		for (int i = 0; i < TRAIN_DATA_VOLUME / BUTCH_SIZE; ++i) {
			std::vector<Matrix*> butch(data.begin() + i * BUTCH_SIZE, data.begin() + (i + 1) * BUTCH_SIZE);
			std::vector<Matrix*> butchOfAnswers(answers.begin() + i * BUTCH_SIZE, answers.begin() + (i + 1) * BUTCH_SIZE);
			makural.optimizerSGD(butch, butchOfAnswers, 0.01, 0.02, 0.0005);
			std::cout << "Butch passed" << std::endl;
		}
		if (TRAIN_DATA_VOLUME % BUTCH_SIZE != 0) {
			std::vector<Matrix*> butch(data.end() - TRAIN_DATA_VOLUME % BUTCH_SIZE, data.end());
			std::vector<Matrix*> butchOfAnswers(answers.end() - TRAIN_DATA_VOLUME % BUTCH_SIZE, answers.end());
			makural.optimizerSGD(butch, butchOfAnswers, 0.01, 0.02, 0.0005);
		}
		std::cout << "Epoch passed" << std::endl;
	}
	
	return 0;
}