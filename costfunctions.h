#ifndef COST_FUNCTIONS_MAKURAL
#define COST_FUNCTIONS_MAKURAL


#include<iostream>

class CostFunction {
public:
	virtual double calculateCost(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const = 0;
};

class CrossEntropy : public CostFunction {
public:
	double calculateCost(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const {
		if (left.rows() != right.rows() || left.cols() != right.cols()) {
			throw "Error in calculateCost";
		}

		double cost = 0.0;
		for (int row = 0; row < left.rows(); ++row) {
			for (int col = 0; col < right.cols(); ++col) {
				if (right(row, col) <= 0) {
					std::cout << right(row, col) << std::endl;
					std::cout << "NAN" << std::endl;
				}
				cost += left(row, col) * std::log(right(row, col));
			}
		}

		return -cost;
	}
};


#endif