#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include <Eigen/Dense>

class ActivationFunction {
protected:
	double alpha_;
public:
	ActivationFunction(double alpha = 0) : alpha_(alpha) {}
	virtual Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const = 0;
	virtual Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const = 0;
};

class Sigmoid : public ActivationFunction {
public:
	Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
};

class ReLu : public ActivationFunction {
public:
	Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
};


class Tanh : public ActivationFunction {
public:
	Eigen::MatrixXd  calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd  calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
};

class ELU : public ActivationFunction {
public:
	ELU(double alpha) : ActivationFunction(alpha) {}
	Eigen::MatrixXd  calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd  calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
};


class Softmax : public ActivationFunction {
public:
	Eigen::MatrixXd  calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd  calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
};


#endif
