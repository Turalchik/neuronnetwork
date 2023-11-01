#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include <Eigen/Dense>

class HiddenActivationFunction {
public:
	virtual Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const = 0;
	virtual Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const = 0;
	virtual const char* getStr() const = 0;
	static HiddenActivationFunction* constructObject(const char* function);
};

class Sigmoid : public HiddenActivationFunction {
public:
	Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
	const char* getStr() const override;
};

class ReLu : public HiddenActivationFunction {
public:
	Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
	const char* getStr() const override;
};

class Tanh : public HiddenActivationFunction {
public:
	Eigen::MatrixXd  calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	Eigen::MatrixXd  calculateDerivativeFunction(const Eigen::MatrixXd& WeightedSums) const override;
	const char* getStr() const override;
};

class OutputActivationFunction {
public:
	virtual Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const = 0;
	virtual const char* getStr() const = 0;
	static OutputActivationFunction* constructObject(const char* function);
};

class Softmax : public OutputActivationFunction {
public:
	Eigen::MatrixXd calculateFunction(const Eigen::MatrixXd& WeightedSums) const override;
	const char* getStr() const override;
};

#endif