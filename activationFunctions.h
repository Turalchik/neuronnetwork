#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include"matrixlab.h"
#include<cmath>
#include<vector>


class ActivationFunction {
protected:
	double alpha_;
public:
	ActivationFunction(double alpha = 0) : alpha_(alpha) {}
	virtual Matrix calculateFunction(const Matrix& WeightedSums) const = 0;
	virtual Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const = 0;
};



class Sigmoid : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = 1.0 / (1.0 + exp(-WeightedSums(i, 0)));
		}
		
		return newMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
		Matrix tempMatrix = ones(WeightedSums.rows(), 1);
		tempMatrix -= ActivatedWeightedSums;
		tempMatrix.elementWiseMultiplication(ActivatedWeightedSums);

		return tempMatrix;
	}
};

class ReLu : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : 0;
		}

		return newMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? 1 : 0;
		}

		return newMatrix;
	
	}
};


class Tanh : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override {
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

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix ActivatedWeightedSums = calculateFunction(WeightedSums);
		Matrix tempMatrix = ones(WeightedSums.rows(), 1);
		tempMatrix -= ActivatedWeightedSums.elementWiseMultiplication(ActivatedWeightedSums);

		return tempMatrix;
	}
};

class ELU : public ActivationFunction {
public:
	ELU(double alpha) : ActivationFunction(alpha) {}
	Matrix calculateFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix tempMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			tempMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : alpha_ * (exp(WeightedSums(i, 0) - 1));
		}

		return tempMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix tempMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			tempMatrix(i, 0) = WeightedSums(i, 0) > 0 ? 1 : alpha_ * (exp(WeightedSums(i, 0)));
		}

		return tempMatrix;
	}
};


class Softmax : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override {
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
};


#endif
