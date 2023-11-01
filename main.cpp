#include"makural.h"
#include "mnistools.h"

int main() {
	NeuralNetwork makural("MNIST.txt");
	Eigen::MatrixXd res = pngToMatrix("pic.png");
	std::cout << makural.predict(res) << " " << makural.answerVec(res).maxCoeff() * 100 << "%";

	return 0;
}