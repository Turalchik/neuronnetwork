#include"makural.h"
#include<fstream>
#include<iostream>

enum CONSTANTS
{
	TRAIN_DATA_VOLUME = 1'000,
	TEST_DATA_VOLUME = 10'000,
	INPUT_NEURON_AMOUNT = 784,
	OUTPUT_NEURON_AMOUNT = 10,
	BUTCH_SIZE = 32,
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
	std::ifstream train("mnist_train_3.csv");
	std::ifstream test("mnist_test.csv");

	std::vector<Matrix*> data_train(TRAIN_DATA_VOLUME);
	std::vector<Matrix*> answers_train(TRAIN_DATA_VOLUME);

	std::vector<Matrix*> data_test(TEST_DATA_VOLUME);
	std::vector<Matrix*> answers_test(TEST_DATA_VOLUME);

	NeuralNetwork makural({784, 128, 32, 10}, new ReLu, new Softmax, new CrossEntropy);

	size_t rowIndex = 0;
	int get = 0;

	while (train.get() != '\n') {}

	std::cout << "Loading dataset..." << std::endl;

	while (!train.eof()) {
		answers_train[rowIndex] = new Matrix(1, OUTPUT_NEURON_AMOUNT);
		(*answers_train[rowIndex])(0, train.get() - '0') = 1.0;
		train.ignore();

		data_train[rowIndex] = new Matrix(1, INPUT_NEURON_AMOUNT);
		for (int i = 0; i < INPUT_NEURON_AMOUNT; ++i) {
			train >> get;
			(*data_train[rowIndex])(0, i) = static_cast<double>(get) / 255.0;
			train.ignore();
		}
		++rowIndex;
		while ((isdigit(train.peek()) == 0) && train.get() != EOF) {}
	}

	rowIndex = 0;

	/*while (test.get() != '\n') {}

	while (!test.eof()) {
		answers_test[rowIndex] = new Matrix(1, OUTPUT_NEURON_AMOUNT);
		(*answers_test[rowIndex])(0, test.get() - '0') = 1.0;
		test.ignore();

		data_test[rowIndex] = new Matrix(1, INPUT_NEURON_AMOUNT);
		for (int i = 0; i < INPUT_NEURON_AMOUNT; ++i) {
			test >> get;
			(*data_test[rowIndex])(0, i) = static_cast<double>(get) / 255.0;
			test.ignore();
		}
		++rowIndex;
		while ((isdigit(test.peek()) == 0) && test.get() != EOF) {}
	}*/

	std::cout << "Dataset was loaded, start training" << std::endl << std::endl;

	makural.train(data_train, answers_train, data_test, answers_test, EPOCHS_AMOUNT, BUTCH_SIZE);


	train.close();
	test.close();

	return 0;
}