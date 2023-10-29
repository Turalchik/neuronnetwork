#include "activationFunctions.h"
#include <iostream>

Eigen::MatrixXd Sigmoid::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd  newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = 1.0 / (1.0 + std::exp(-WeightedSums(i, 0)));
	}

	return newMatrix;
}

Eigen::MatrixXd  Sigmoid::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd  ActivatedWeightedSums = calculateFunction(WeightedSums);
	Eigen::MatrixXd  tempMatrix = Eigen::MatrixXd::Ones(1, WeightedSums.cols());
	tempMatrix -= ActivatedWeightedSums;

	return tempMatrix.cwiseProduct(ActivatedWeightedSums);;
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

Eigen::MatrixXd ELU::calculateFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd tempMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		tempMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? WeightedSums(0, i) : alpha_ * (std::exp(WeightedSums(0, i) - 1.0));
	}

	return tempMatrix;
}

Eigen::MatrixXd ELU::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Eigen::MatrixXd tempMatrix(1, WeightedSums.cols());
	for (size_t i = 0; i < WeightedSums.cols(); ++i) {
		tempMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? 1.0 : alpha_ * (std::exp(WeightedSums(0, i)));
	}

	return tempMatrix;
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

Eigen::MatrixXd Softmax::calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const {
	// UNSOPPORTED
	return Eigen::MatrixXd();
}