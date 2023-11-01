#include "costfunctions.h"

CostFunction* CostFunction::constructObject(const char* function) {
	if (!_strcmpi(function, "crossentropy")) {
		return new CrossEntropy;
	}
	throw "There's no cost function like this.";
}

const char* CrossEntropy::getStr() const {
	return "crossentropy";
}

double CrossEntropy::calculateCost(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const {
	if (left.rows() != right.rows() || left.cols() != right.cols()) {
		throw "Error in calculateCost";
	}

	double cost = 0.0;
	for (int row = 0; row < left.rows(); ++row) {
		for (int col = 0; col < right.cols(); ++col) {
			cost += left(row, col) * std::log(right(row, col));
		}
	}

	return -cost;
}