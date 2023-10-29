#include"makural.h"
#include<fstream>
#include<string>

enum CONSTANTS
{
	INPUT_NEURON_AMOUNT = 784,
	OUTPUT_NEURON_AMOUNT = 10,
	BUTCH_SIZE = 32,
	EPOCHS_AMOUNT = 20,
};

void MNISTLoader(std::ifstream& data_file, std::vector<Eigen::MatrixXd*>& data_vect,
				 std::vector<Eigen::MatrixXd*>& answers_vect, const char* filename = "data") {

	std::cout << "Loading " << filename << " file..." << std::endl << std::endl;

	Eigen::MatrixXd* tempMatrix;
	int get = 0;
	while (data_file.get() != '\n') {}

	while (!data_file.eof()) {
		tempMatrix = new Eigen::MatrixXd(1, OUTPUT_NEURON_AMOUNT);
		tempMatrix->setZero();
		answers_vect.push_back(tempMatrix);
		(*answers_vect[answers_vect.size() - 1])(0, data_file.get() - '0') = 1.0;
		data_file.ignore();

		data_vect.push_back(new Eigen::MatrixXd(1, INPUT_NEURON_AMOUNT));
		for (int i = 0; i < INPUT_NEURON_AMOUNT; ++i) {
			data_file >> get;
			(*data_vect[data_vect.size() - 1])(0, i) = static_cast<double>(get) / 255.0;
			data_file.ignore();
		}
		while ((isdigit(data_file.peek()) == 0) && data_file.get() != EOF) {}
	}

	std::cout << filename << " file was loaded." << std::endl << std::endl;
}


int main() {
	const char* train_data_path = "mnist_train.csv";
	const char* test_data_path = "mnist_test.csv";

	std::ifstream train_file(train_data_path);
	std::ifstream test_file(test_data_path);

	std::vector<Eigen::MatrixXd*> data_train;
	std::vector<Eigen::MatrixXd*> answers_train;
	std::vector<Eigen::MatrixXd*> data_test;
	std::vector<Eigen::MatrixXd*> answers_test;

	MNISTLoader(train_file, data_train, answers_train, train_data_path);
	MNISTLoader(test_file, data_test, answers_test, test_data_path);

	NeuralNetwork makural({784, 128, 32, 10 }, new ReLu, new Softmax, new CrossEntropy);
	makural.train(data_train, answers_train, data_test, answers_test, EPOCHS_AMOUNT, BUTCH_SIZE);
	makural.save(std::to_string(data_train.size() / 1000) + "k_train_examples_acc_" +
				 std::to_string(makural.calculateAccuracy(data_test, answers_test)) + ".txt");
	std::cout << makural.calculateAccuracy(data_train, answers_train) << std::endl;

	train_file.close();
	test_file.close();

	return 0;
}