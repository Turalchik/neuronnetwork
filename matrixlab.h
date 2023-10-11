#ifndef MATRIXFORMATRIXEXP
#define MATRIXFORMATRIXEXP

#include<fstream>
#include "rational.h"

template<typename T>
class Matrix {
	size_t rows_;
	size_t columns_;
	T** matrix_;

	void swap(Matrix& other) {
		std::swap(rows_, other.rows_);
		std::swap(columns_, other.columns_);
		std::swap(matrix_, other.matrix_);
	}

public:
	Matrix(const size_t rows, const size_t columns);
	Matrix(const T& X);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;
	~Matrix();
	Matrix& operator= (const Matrix& other);
	Matrix& operator= (Matrix&& other) noexcept;

	size_t rows() const;
	size_t columns() const;
	size_t size() const;

	Matrix operator- () const;
	Matrix operator+ () const;

	Matrix& operator+= (const Matrix& other);
	Matrix& operator-= (const Matrix& other);
	Matrix& operator*= (const Matrix& other);
	Matrix& operator/= (const Matrix& other);

	T& operator() (size_t i, size_t j);
	T*& operator() (size_t i) const;
	const T& operator() (size_t i, size_t j) const;
};

template<typename T>
Matrix<T> operator+ (const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
Matrix<T> operator- (const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
Matrix<T> operator* (const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
Matrix<T> operator/ (const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
Matrix<T> transpose(const Matrix<T>& X);

template<typename T>
Matrix<T> invMatrix(const Matrix<T>& X);

template<typename T>
Matrix<T> det(const Matrix<T>& X);

template<typename T>
Matrix<T> zeros(const Matrix<T>& rows, const Matrix<T>& columns);

template<typename T>
Matrix<T> eye(const Matrix<T>& n);

template<typename T>
Matrix<T> elementWiseMultiplication(const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
Matrix<T> elementWiseDivision(const Matrix<T>& X, const Matrix<T>& Y);

template<typename T>
bool operator== (const Matrix<T>& x, const Matrix<T>& y);

template<typename T>
bool operator!= (const Matrix<T>& x, const Matrix<T>& y);


#endif