#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include"matrixlab.h"
#include<cmath>


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
	Matrix calculateFunction(const Matrix& WeightedSums) const override;
	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override;
};

class ReLu : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override;
	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override;
};


class Tanh : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override;
	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override;
};

class ELU : public ActivationFunction {
public:
	ELU(double alpha) : ActivationFunction(alpha) {}
	Matrix calculateFunction(const Matrix& WeightedSums) const override;
	Matrix calculateDerivativeFunction(const Matrix& WeightedSums) const override;
};


class Softmax : public ActivationFunction {
public:
	Matrix calculateFunction(const Matrix& WeightedSums) const override;
};


#endif
