#include "activationFunctions.h"
#include <iostream>

HiddenActivationFunction* HiddenActivationFunction::constructObject(const char* function) {
	if (!_strcmpi(function, "sigmoid")) {
		return new Sigmoid;
	}
	if (!_strcmpi(function, "relu")) {
		return new ReLu;
	}
	if (!_strcmpi(function, "tanh")) {
		return new Tanh;
	}
	throw "There's no hidden layer activation function like this.";
}

Eigen::MatrixXd Sigmoid::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd  newMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		newMatrix(0, i) = 1.0 / (1.0 + std::exp(-WeightedSums(0, i)));
	}

	return newMatrix;
}

Eigen::MatrixXd Sigmoid::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd  ActivatedWeightedSums = calculateFunction(WeightedSums);
	Eigen::MatrixXd  tempMatrix = Eigen::MatrixXd::Ones(1, WeightedSums.cols());
	tempMatrix -= ActivatedWeightedSums;

	return tempMatrix.cwiseProduct(ActivatedWeightedSums);;
}

const char* Sigmoid::getStr() const {
	return "sigmoid";
}

Eigen::MatrixXd ReLu::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd newMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		newMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? WeightedSums(0, i) : 0;
	}

	return newMatrix;
}

Eigen::MatrixXd ReLu::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd newMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		newMatrix(0, i) = (WeightedSums(0, i) > 0) ? 1 : 0;
	}

	return newMatrix;

}

const char* ReLu::getStr() const {
	return "relu";
}

Eigen::MatrixXd Tanh::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd newMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(0, i) = (std::exp(WeightedSums(0, i)) - std::exp(-WeightedSums(0, i))) /
			(std::exp(WeightedSums(0, i)) + std::exp(-WeightedSums(0, i)));
	}

	return newMatrix;
}

Eigen::MatrixXd Tanh::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd ActivatedWeightedSums = calculateFunction(WeightedSums);
	Eigen::MatrixXd tempMatrix = Eigen::MatrixXd::Ones(1, WeightedSums.cols());
	return tempMatrix -= ActivatedWeightedSums.cwiseProduct(ActivatedWeightedSums);
}

const char* Tanh::getStr() const {
	return "tanh";
}

OutputActivationFunction* OutputActivationFunction::constructObject(const char* function) {
	if (!_strcmpi(function, "softmax")) {
		return new Softmax;
	}
	throw "There's no output layer activation function like this.";
}

Eigen::MatrixXd Softmax::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	double LowerSum = 0;
	double ExponentedElement = 0;
	Eigen::MatrixXd tempMatrix(1, WeightedSums.cols());

	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		ExponentedElement = std::exp(WeightedSums(0, i));
		tempMatrix(0, i) = ExponentedElement;
		LowerSum += ExponentedElement;
	}

	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		tempMatrix(0, i) /= LowerSum;
	}

	return tempMatrix;
}

const char* Softmax::getStr() const {
	return "softmax";
}