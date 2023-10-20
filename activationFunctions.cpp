#include"activationFunctions.h"

Matrix Sigmoid::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = 1.0 / (1.0 + exp(-WeightedSums(i, 0)));
	}

	return newMatrix;
}

Matrix Sigmoid::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
	Matrix tempMatrix = ones(WeightedSums.rows(), 1);
	tempMatrix -= ActivatedWeightedSums;
	tempMatrix.elementWiseMultiplication(ActivatedWeightedSums);

	return tempMatrix;
}

Matrix ReLu::calculateFunction(const Matrix& WeightedSums) const  {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : 0;
	}

	return newMatrix;
}

Matrix ReLu::calculateDerivativeFunction(const Matrix& WeightedSums) const  {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? 1 : 0;
	}

	return newMatrix;

}

Matrix Tanh::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = (exp(WeightedSums(i, 0)) - exp(-WeightedSums(i, 0))) /
			(exp(WeightedSums(i, 0)) + exp(-WeightedSums(i, 0)));
	}

	return newMatrix;
}

Matrix Tanh::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
	Matrix tempMatrix = ones(WeightedSums.rows(), 1);
	tempMatrix -= ActivatedWeightedSums.elementWiseMultiplication(ActivatedWeightedSums);

	return tempMatrix;
}

ELU::ELU(double alpha) : ActivationFunction(alpha) {}

Matrix ELU::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix tempMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		tempMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : alpha_ * (exp(WeightedSums(i, 0) - 1));
	}

	return tempMatrix;
}

Matrix ELU::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix tempMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		tempMatrix(i, 0) = WeightedSums(i, 0) > 0 ? 1 : alpha_ * (exp(WeightedSums(i, 0)));
	}

	return tempMatrix;
}

Matrix Softmax::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.columns() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	double LowerSum = 0;
	double ExponentedElement = 0;
	Matrix tempMatrix(WeightedSums.rows(), 1);

	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		ExponentedElement = exp(WeightedSums(i, 0));
		tempMatrix(i, 0) = ExponentedElement;
		LowerSum += ExponentedElement;
	}

	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		tempMatrix(i, 0) /= LowerSum;
	}

	return tempMatrix;
}