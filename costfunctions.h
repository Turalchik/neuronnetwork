#ifndef COST_FUNCTIONS_MAKURAL
#define COST_FUNCTIONS_MAKURAL

#include<iostream>
#include<Eigen/Dense>

class CostFunction {
public:
	virtual double calculateCost(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const = 0;
	virtual const char* getStr() const = 0;
	static CostFunction* constructObject(const char* function);
};

class CrossEntropy : public CostFunction {
public:
	double calculateCost(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const override;
	const char* getStr() const override;
};

#endif