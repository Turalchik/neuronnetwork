#include"activationFunctions.h"

Matrix Sigmoid::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(WeightedSums.rows(), 1);
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(i, 0) = 1.0 / (1.0 + exp(-WeightedSums(i, 0)));
	}

	return newMatrix;
}

Matrix Sigmoid::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
	Matrix tempMatrix = ones(1, WeightedSums.columns());
	tempMatrix -= ActivatedWeightedSums;
	tempMatrix.elementWiseMultiplication(ActivatedWeightedSums);

	return tempMatrix;
}

Matrix ReLu::calculateFunction(const Matrix& WeightedSums) const  {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(1, WeightedSums.columns());
	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		newMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? WeightedSums(0, i) : 0;
	}

	return newMatrix;
}

Matrix ReLu::calculateDerivativeFunction(const Matrix& WeightedSums) const  {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(1, WeightedSums.columns());
	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		newMatrix(0, i) = (WeightedSums(0, i) > 0) ? 1 : 0;
	}

	return newMatrix;

}

Matrix Tanh::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix newMatrix(1, WeightedSums.columns());
	for (size_t i = 0; i < WeightedSums.rows(); ++i) {
		newMatrix(0, i) = (exp(WeightedSums(0, i)) - exp(-WeightedSums(0, i))) /
			(exp(WeightedSums(0, i)) + exp(-WeightedSums(0, i)));
	}

	return newMatrix;
}

Matrix Tanh::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
	Matrix tempMatrix = ones(1, WeightedSums.columns());
	tempMatrix -= ActivatedWeightedSums.elementWiseMultiplication(ActivatedWeightedSums);

	return tempMatrix;
}

Matrix ELU::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix tempMatrix(1, WeightedSums.columns());
	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		tempMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? WeightedSums(0, i) : alpha_ * (exp(WeightedSums(0, i) - 1.0));
	}

	return tempMatrix;
}

Matrix ELU::calculateDerivativeFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	Matrix tempMatrix(1, WeightedSums.columns());
	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		tempMatrix(0, i) = (WeightedSums(0, i) > 0.0) ? 1.0 : alpha_ * (exp(WeightedSums(0, i)));
	}

	return tempMatrix;
}

Matrix Softmax::calculateFunction(const Matrix& WeightedSums) const {
	if (WeightedSums.rows() != 1) {
		throw "Wrong WeightedSums vector size.";
	}

	double LowerSum = 0;
	double ExponentedElement = 0;
	Matrix tempMatrix(1, WeightedSums.columns());

	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		ExponentedElement = exp(WeightedSums(0, i));
		tempMatrix(0, i) = ExponentedElement;
		LowerSum += ExponentedElement;
	}

	for (size_t i = 0; i < WeightedSums.columns(); ++i) {
		tempMatrix(0, i) /= LowerSum;
	}

	return tempMatrix;
}