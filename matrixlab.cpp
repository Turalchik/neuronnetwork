#include "matrixlab.h"
#include<iostream>
#include<random>

Matrix::Matrix(const size_t& rows, const size_t& columns) : rows_(rows), columns_(columns) {
	matrix_ = new double* [rows_];
	for (size_t i = 0; i < rows_; ++i) {
		matrix_[i] = new double[columns_];
	}
}


Matrix::Matrix(const double& X) : rows_(1), columns_(1) {
	matrix_ = new double* [rows_];
	matrix_[0] = new double[columns_];
	matrix_[0][0] = X;
}


Matrix::Matrix(const Matrix& other) : Matrix(other.rows_, other.columns_) {
	for (size_t i = 0; i < other.rows_; ++i) {
		for (size_t j = 0; j < other.columns_; ++j) {
			matrix_[i][j] = other.matrix_[i][j];
		}
	}
}

Matrix Matrix::multiplicationByTransposeMatrix(const Matrix& other) const {
	if (columns_ != other.columns_) {
		throw "Incorrect size";
	}

	Matrix result(rows_, other.rows_);
	double tempNumber;

	for (int row = 0; row < result.rows_; ++row) {
		for (int col = 0; col < result.columns_; ++col) {
			tempNumber = 0.0;
			for (int gen = 0; gen < columns_; ++gen) {
				tempNumber += matrix_[row][gen] * other.matrix_[col][gen];
			}
			result.matrix_[row][col] = tempNumber;
		}
	}

	return result;
}

Matrix Matrix::multiplicationTransposeByMatrix(const Matrix& other) const {
	if (rows_ != other.rows_) {
		throw "Incorrect size";
	}

	Matrix result(columns_, other.columns_);
	double tempNumber;

	for (int row = 0; row < result.rows_; ++row) {
		for (int col = 0; col < result.columns_; ++col) {
			tempNumber = 0.0;
			for (int gen = 0; gen < rows_; ++gen) {
				tempNumber += matrix_[gen][row] * other.matrix_[gen][col];
			}
			result.matrix_[row][col] = tempNumber;
		}
	}

	return result;
}


Matrix::Matrix(Matrix&& other) noexcept {
	matrix_ = nullptr;
	rows_ = 0;
	columns_ = 0;
	swap(other);
}


Matrix::~Matrix() {
	free();
}


Matrix& Matrix::operator= (const Matrix& other) {
	if (this != &other) {
		Matrix tempObj(other);
		swap(tempObj);
	}
	return *this;
}


Matrix& Matrix::operator= (Matrix&& other) noexcept {
	if (this != &other) {
		swap(other);
	}
	return *this;
}


Matrix zeros(const size_t& rows, const size_t& columns) {
	Matrix tempObj(rows, columns);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < columns; ++j) {
			tempObj(i, j) = 0;
		}
	}
	return tempObj;
}

Matrix ones(const size_t& rows, const size_t& columns) {
	Matrix tempObj(rows, columns);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < columns; ++j) {
			tempObj(i, j) = 1;
		}
	}
	return tempObj;
}


Matrix eye(const size_t& n) {
	Matrix tempObj(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			tempObj(i, j) = (i == j) ? 1 : 0;
		}
	}
	return tempObj;
}


size_t Matrix::rows() const {
	return rows_;
}


size_t Matrix::columns() const {
	return columns_;
}


size_t Matrix::size() const {
	return rows_ * columns_;
}

void Matrix::free() {
	for (size_t i = 0; i < rows_; ++i) {
		delete[] matrix_[i];
	}
	delete[] matrix_;
}

Matrix Matrix::operator- () const {
	Matrix tempObj(*this);
	tempObj *= (-1);
	return tempObj;
}


Matrix Matrix::operator+ () const {
	return *this;
}


Matrix& Matrix::operator+= (const Matrix& other) {
	if (rows_ == other.rows_ && columns_ == other.columns_) {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < columns_; ++j) {
				matrix_[i][j] += other.matrix_[i][j];
			}
		}
		return *this;
	}
	throw "Matrices of different sizes cannot be added.";
}


Matrix& Matrix::operator-= (const Matrix& other) {
	return *this += (-other);
}


Matrix& Matrix::operator*= (const Matrix& other) {
	if (columns_ == other.rows_) {
		double result;
		Matrix tempObj(rows_, other.columns_);
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < other.columns_; ++j) {
				result = 0;
				for (size_t k = 0; k < columns_; ++k) {
					result += matrix_[i][k] * other.matrix_[k][j];
				}
				tempObj.matrix_[i][j] = result;
			}
		}
		swap(tempObj);
	}
	else if (other.rows_ == 1 && other.columns_ == 1) {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < columns_; ++j) {
				matrix_[i][j] *= other.matrix_[0][0];
			}
		}
	}
	else if (rows_ == 1 && columns_ == 1) {
		Matrix tempObj(other);
		for (size_t i = 0; i < tempObj.rows_; ++i) {
			for (size_t j = 0; j < tempObj.columns_; ++j) {
				tempObj.matrix_[i][j] *= matrix_[0][0];
			}
		}
		swap(tempObj);
	}
	else {
		throw "Matrix multiplication is not possible.";
	}

	return *this;

}


Matrix& Matrix::operator/= (const Matrix& other) {
	if (other.rows_ == 1 && other.columns_ == 1) {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < columns_; ++j) {
				matrix_[i][j] /= other.matrix_[0][0];
			}
		}
	}
	else {
		*this *= invMatrix(other);
	}

	return *this;
}


double& Matrix::operator() (const size_t& i, const size_t& j) {
	if (i < rows_ && j < columns_) {
		return matrix_[i][j];
	}
	throw "Out of bounds.";
}


double*& Matrix::operator() (const size_t& i) const {
	return matrix_[i];
}


const double& Matrix::operator() (const size_t& i, const size_t& j) const {
	if (i < rows_ && j < columns_) {
		return matrix_[i][j];
	}
	throw "Out of bounds.";
}

Matrix& Matrix::elementWiseMultiplication(const Matrix& other) {
	if (rows_ == other.rows_ && columns_ == other.columns_) {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < columns_; ++j) {
				matrix_[i][j] *= other.matrix_[i][j];
			}
		}

		return *this;
	}
	throw "Element-wise multiplication is not possible.";
}

Matrix Matrix::elementWiseMultiplicationTransposeByMatrix(Matrix&& other) const {
	if (columns_ == other.rows_ && rows_ == other.columns_) {
		for (size_t i = 0; i < columns_; ++i) {
			for (size_t j = 0; j < rows_; ++j) {
				other.matrix_[i][j] *= matrix_[j][i];
			}
		}

		return other;
	}
	throw "Element-wise multiplication is not possible.";
}

Matrix& Matrix::elementWiseDivision(const Matrix& other) {
	if (rows_ == other.rows_ && columns_ == other.columns_) {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < columns_; ++j) {
				matrix_[i][j] /= other.matrix_[i][j];
			}
		}

		return *this;
	}
	throw "Element-wise multiplication is not possible.";
}

void Matrix::FillMatrixByRandomNumbers(const double& afterActivationSize) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dis(0.0, std::sqrt(2.0 / afterActivationSize));

	for (size_t i = 0; i < rows_; ++i) {
		for (size_t j = 0; j < columns_; ++j) {
			matrix_[i][j] = dis(gen);
		}
	}
}

void Matrix::fillWithZeros() {
	for (size_t i = 0; i < rows_; ++i) {
		for (size_t j = 0; j < columns_; ++j) {
			matrix_[i][j] = 0.0;
		}
	}
}

double Matrix::minElement() const {
	double min_element = matrix_[0][0];
	for (size_t i = 0; i < rows_; ++i) {
		for (size_t j = 0; j < columns_; ++j) {
			if (matrix_[i][j] < min_element) {
				min_element = matrix_[i][j];
			}
		}
	}
	return min_element;
}

double Matrix::maxElement() const {
	double max_element = matrix_[0][0];
	for (size_t i = 0; i < rows_; ++i) {
		for (size_t j = 0; j < columns_; ++j) {
			if (matrix_[i][j] > max_element) {
				max_element = matrix_[i][j];
			}
		}
	}
	return max_element;
}

Matrix operator+ (const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj += Y;
	return tempObj;
}


Matrix operator- (const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj -= Y;
	return tempObj;
}


Matrix operator* (const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj *= Y;
	return tempObj;
}


Matrix operator/ (const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj /= Y;
	return tempObj;
}


Matrix transpose(const Matrix& X) {
	size_t rows = X.rows();
	size_t columns = X.columns();

	Matrix tempObj(columns, rows);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < columns; ++j) {
			tempObj(j, i) = X(i, j);
		}
	}
	return tempObj;
}


Matrix invMatrix(const Matrix& X) {

	if (X.rows() == X.columns()) {
		Matrix eyeMatrix = eye(X.rows());
		Matrix tempObj(X);
		double tempValue;
		size_t rowIndex;

		for (size_t diagonalIndex = 0; diagonalIndex < tempObj.rows(); ++diagonalIndex) {

			rowIndex = diagonalIndex;
			while (rowIndex < tempObj.rows() && tempObj(rowIndex, diagonalIndex) == 0) {
				++rowIndex;
			}

			if (rowIndex == tempObj.rows()) {
				throw "Determinant equals to zero.";
			}

			if (rowIndex != diagonalIndex) {
				std::swap(tempObj(diagonalIndex), tempObj(rowIndex));
				std::swap(eyeMatrix(diagonalIndex), eyeMatrix(rowIndex));
			}

			for (size_t i = diagonalIndex + 1; i < tempObj.rows(); ++i) {
				tempValue = tempObj(i, diagonalIndex) / tempObj(diagonalIndex, diagonalIndex);
				for (size_t j = 0; j < tempObj.columns(); ++j) {

					tempObj(i, j) -= tempObj(diagonalIndex, j) * tempValue;
					eyeMatrix(i, j) -= eyeMatrix(diagonalIndex, j) * tempValue;
				}
			}
		}

		for (size_t j = tempObj.columns() - 1; j > 0; --j) {
			for (size_t i = 0; i < j; ++i) {
				tempValue = tempObj(i, j) / tempObj(j, j);
				tempObj(i, j) -= tempObj(j, j) * tempValue;
				for (size_t eyeIndex = 0; eyeIndex < eyeMatrix.columns(); ++eyeIndex) {
					eyeMatrix(i, eyeIndex) -= eyeMatrix(j, eyeIndex) * tempValue;
				}
			}
		}

		for (size_t i = 0; i < eyeMatrix.rows(); ++i) {
			for (size_t j = 0; j < eyeMatrix.columns(); ++j) {
				eyeMatrix(i, j) /= tempObj(i, i);
			}
		}

		return eyeMatrix;
	}

	throw "Inverse matrix can not be calculated.";
}

Matrix elementWiseMultiplication(const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj.elementWiseMultiplication(Y);
	return tempObj;
}

Matrix elementWiseDivision(const Matrix& X, const Matrix& Y) {
	Matrix tempObj(X);
	tempObj.elementWiseDivision(Y);
	return tempObj;
}


bool operator== (const Matrix& X, const Matrix& Y) {
	if (X.rows() == Y.rows() && X.columns() == Y.columns()) {
		size_t rows = X.rows();
		size_t columns = X.columns();
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < columns; ++j) {
				if (X(i, j) != Y(i, j)) {
					return false;
				}
			}
		}
		return true;
	}
	return false;
}


bool operator!= (const Matrix& X, const Matrix& Y) {
	return !(X == Y);
}
