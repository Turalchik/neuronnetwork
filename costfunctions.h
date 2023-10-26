#ifndef COST_FUNCTIONS_MAKURAL
#define COST_FUNCTIONS_MAKURAL


#include "matrixlab.h"
#include<iostream>

class CostFunction {
public:
	virtual double calculateCost(const Matrix& left, const Matrix& right) const = 0;
};

class CrossEntropy : public CostFunction {
public:
	double calculateCost(const Matrix& left, const Matrix& right) const {
		if (left.rows() != right.rows() || left.columns() != right.columns()) {
			throw "Error in calculateCost";
		}

		double cost = 0.0;
		for (int row = 0; row < left.rows(); ++row) {
			for (int col = 0; col < right.columns(); ++col) {
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