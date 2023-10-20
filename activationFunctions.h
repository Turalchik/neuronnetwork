#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include"matrixlab.h"
#include<cmath>


class ActivationFunction {
protected:
	double alpha_;
public:
	ActivationFunction(double alpha) : alpha_(alpha) {}
	virtual Matrix calculateFunction(const Matrix& WeightedSums) = 0;
	virtual Matrix calculateDerivativeFunction(const Matrix& WeightedSums) = 0;
};



class Sigmoid : public ActivationFunction {
public:
	Sigmoid(double alpha = 0) : ActivationFunction(alpha) {}
	Matrix calculateFunction(const Matrix& WeightedSums) override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = 1.0 / (1.0 + exp(-WeightedSums(i, 0)));
		}
		
		return newMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) override {
		return calculateFunction(WeightedSums) * (1 - calculateFunction(WeightedSums));
	}
};

class ReLu : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : 0;
		}

		return newMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) override {
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
	Matrix calculateFunction(const Matrix& WeightedSums) override {
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

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) override {
		return 1 - calculateFunction(WeightedSums) * calculateFunction(WeightedSums);
	}
};

class ELU : public ActivationFunction {
public:
	ELU(double alpha) : ActivationFunction(alpha) {}
	Matrix calculateFunction(const Matrix& WeightedSums) override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? WeightedSums(i, 0) : alpha_ * (exp(WeightedSums(i, 0) - 1));
		}

		return newMatrix;
	}

	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) > 0 ? 1 : alpha_ * (exp(WeightedSums(i, 0)));
		}

		return newMatrix;

	}
};


class Softmax : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) override {
		if (WeightedSums.columns() != 1) {
			throw "Wrong WeightedSums vector size.";
		}
		
		double LowerSum = 0;
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			LowerSum += exp(WeightedSums(i, 0));
		}

		Matrix newMatrix(WeightedSums.rows(), 1);
		for (size_t i = 0; i < WeightedSums.rows(); ++i) {
			newMatrix(i, 0) = WeightedSums(i, 0) / LowerSum;
		}
	}
};


#endif
