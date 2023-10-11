#include "matrixlab.h"

template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t columns) : rows_(rows), columns_(columns) {
	matrix_ = new T * [rows_];
	for (size_t i = 0; i < rows_; ++i) {
		matrix_[i] = new T[columns_];
	}
}

template<typename T>
Matrix<T>::Matrix(const T& X) {
	matrix_[0][0] = X;
}

template<typename T>
Matrix<T>::Matrix(const Matrix& other) : Matrix(other.rows_, other.columns_) {
	for (size_t i = 0; i < other.rows_; ++i) {
		for (size_t j = 0; j < other.columns_; ++j) {
			matrix_[i][j] = other.matrix_[i][j];
		}
	}
}

template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
	matrix_ = nullptr;
	rows_ = 0;
	columns_ = 0;
	swap(other);
}

template<typename T>
Matrix<T>::~Matrix() {
	for (size_t i = 0; i < rows_; ++i) {
		delete[] matrix_[i];
	}
	delete[] matrix_;
}

template<typename T>
Matrix<T>& Matrix<T>::operator= (const Matrix& other) {
	if (this != &other) {
		Matrix tempObj(other);
		swap(tempObj);
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator= (Matrix&& other) noexcept {
	if (this != &other) {
		swap(other);
	}
	return *this;
}

template<typename T>
Matrix<T> zeros(const size_t rows, const size_t columns) {
	Matrix tempObj(rows, columns);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < columns; ++j) {
			tempObj(i, j) = 0;
		}
	}
	return tempObj;
}

template<typename T>
Matrix<T> eye(const size_t n) {
	Matrix tempObj(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			tempObj(i, j) = (i == j) ? 1 : 0;
		}
	}
	return tempObj;
}

template<typename T>
size_t Matrix<T>::rows() const {
	return rows_;
}

template<typename T>
size_t Matrix<T>::columns() const {
	return columns_;
}

template<typename T>
size_t Matrix<T>::size() const {
	return rows_ * columns_;
}

template<typename T>
Matrix<T> Matrix<T>::operator- () const {
	Matrix tempObj(*this);
	tempObj *= (-1);
	return tempObj;
}

template<typename T>
Matrix<T> Matrix<T>::operator+ () const {
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+= (const Matrix& other) {
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

template<typename T>
Matrix<T>& Matrix<T>::operator-= (const Matrix& other) {
	return *this += (-other);
}

template<typename T>
Matrix<T>& Matrix<T>::operator*= (const Matrix& other) {
	if (columns_ == other.rows_) {
		T result;
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

template<typename T>
Matrix<T>& Matrix<T>::operator/= (const Matrix& other) {
	*this *= invMatrix(other);
	return *this;
}

template<typename T>
T& Matrix<T>::operator() (size_t i, size_t j) {
	if (i < rows_ && j < columns_) {
		return matrix_[i][j];
	}
	throw "Out of bounds.";
}

template<typename T>
T*& Matrix<T>::operator() (size_t i) const {
	return matrix_[i];
}

template<typename T>
const T& Matrix<T>::operator() (size_t i, size_t j) const {
	if (i < rows_ && j < columns_) {
		return matrix_[i][j];
	}
	throw "Out of bounds.";
}

template<typename T>
Matrix<T> operator+ (const Matrix<T>& X, const Matrix<T>& Y) {
	Matrix tempObj(X);
	tempObj += Y;
	return tempObj;
}

template<typename T>
Matrix<T> operator- (const Matrix<T>& X, const Matrix<T>& Y) {
	Matrix tempObj(X);
	tempObj -= Y;
	return tempObj;
}

template<typename T>
Matrix<T> operator* (const Matrix<T>& X, const Matrix<T>& Y) {
	Matrix tempObj(X);
	tempObj *= Y;
	return tempObj;
}

template<typename T>
Matrix<T> operator/ (const Matrix<T>& X, const Matrix<T>& Y) {
	Matrix tempObj(X);
	tempObj /= Y;
	return tempObj;
}

template<typename T>
Matrix<T> transpose(const Matrix<T>& X) {
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

template<typename T>
Matrix<T> invMatrix(const Matrix<T>& X) {

	if (X.rows() == X.columns()) {
		Matrix eyeMatrix = eye(X.rows());
		Matrix tempObj(X);
		T tempValue;
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

template<typename T>
Matrix<T> det(const Matrix<T>& X) {

	if (X.rows() == X.columns()) {
		Matrix tempObj(X);
		T tempValue;
		size_t rowIndex;
		bool detSign = false;

		for (size_t diagonalIndex = 0; diagonalIndex < tempObj.rows(); ++diagonalIndex) {

			rowIndex = diagonalIndex;
			while (rowIndex < tempObj.rows() && tempObj(rowIndex, diagonalIndex) == 0) {
				++rowIndex;
			}

			if (rowIndex == tempObj.rows()) {
				continue;
			}

			if (rowIndex != diagonalIndex) {
				detSign = !detSign;
				std::swap(tempObj(diagonalIndex), tempObj(rowIndex));
			}

			for (size_t i = diagonalIndex + 1; i < tempObj.rows(); ++i) {
				tempValue = tempObj(i, diagonalIndex) / tempObj(diagonalIndex, diagonalIndex);
				for (size_t j = diagonalIndex; j < tempObj.columns(); ++j) {
					tempObj(i, j) -= tempObj(diagonalIndex, j) * tempValue;
				}
			}
		}

		tempValue = 1;
		for (size_t diagonalIndex = 0; diagonalIndex < tempObj.rows(); ++diagonalIndex) {
			tempValue *= tempObj(diagonalIndex, diagonalIndex);
		}

		return (detSign ? Matrix<T>(-tempValue) : Matrix<T>(tempValue));
	}
	throw "Determinant of non-square matrix can't be calculated.";
}

template<typename T>
Matrix<T> elementWiseMultiplication(const Matrix<T>& X, const Matrix<T>& Y) {
	if (X.rows() == Y.rows() && X.columns() == Y.columns()) {
		size_t rows = X.rows();
		size_t columns = X.columns();
		Matrix tempObj(rows, columns);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < columns; ++j) {
				tempObj(i, j) = X(i, j) * Y(i, j);
			}
		}

		return tempObj;
	}
	throw "Element-wise multiplication is not possible.";
}

template<typename T>
Matrix<T> elementWiseDivision(const Matrix<T>& X, const Matrix<T>& Y) {
	if (X.rows() == Y.rows() && X.columns() == Y.columns()) {
		size_t rows = X.rows();
		size_t columns = X.columns();
		Matrix tempObj(rows, columns);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < columns; ++j) {
				tempObj(i, j) = X(i, j) / Y(i, j);
			}
		}

		return tempObj;
	}
	throw "Element-wise division is not possible.";
}

template<typename T>
bool operator== (const Matrix<T>& X, const Matrix<T>& Y) {
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

template<typename T>
bool operator!= (const Matrix<T>& X, const Matrix<T>& Y) {
	return !(X == Y);
}
