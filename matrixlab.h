#ifndef MATRIXFORMATRIXEXP
#define MATRIXFORMATRIXEXP

#include<fstream>
#include<string>

class Matrix {
	size_t rows_;
	size_t columns_;
	double** matrix_;

	void swap(Matrix& other) {
		std::swap(rows_, other.rows_);
		std::swap(columns_, other.columns_);
		std::swap(matrix_, other.matrix_);
	}

public:
	Matrix(const size_t& rows, const size_t& columns);
	Matrix(const double& X);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;
	~Matrix();
	Matrix& operator= (const Matrix& other);
	Matrix& operator= (Matrix&& other) noexcept;

	size_t rows() const;
	size_t columns() const;
	size_t size() const;

	void free();

	Matrix operator- () const;
	Matrix operator+ () const;

	Matrix& operator+= (const Matrix& other);
	Matrix& operator-= (const Matrix& other);
	Matrix& operator*= (const Matrix& other);
	Matrix& operator/= (const Matrix& other);

	double& operator() (const size_t& i, const size_t& j);
	double*& operator() (const size_t& i) const;
	const double& operator() (const size_t& i, const size_t& j) const;

	Matrix& elementWiseMultiplication(const Matrix& other);
	Matrix& elementWiseDivision(const Matrix& other);
	void FillMatrixByRandomNumbers(const double& begin, const double& end);
	double minElement() const;
	double maxElement() const;
};

Matrix operator+ (const Matrix& X, const Matrix& Y);
Matrix operator- (const Matrix& X, const Matrix& Y);
Matrix operator* (const Matrix& X, const Matrix& Y);
Matrix operator/ (const Matrix& X, const Matrix& Y);
Matrix transpose(const Matrix& X);
Matrix invMatrix(const Matrix& X);
Matrix zeros(const size_t& rows, const size_t& columns);
Matrix eye(const size_t& n);
Matrix ones(const size_t& rows, const size_t& columns);
Matrix elementWiseMultiplication(const Matrix& X, const Matrix& Y);
Matrix elementWiseDivision(const Matrix& X, const Matrix& Y);
bool operator== (const Matrix& x, const Matrix& y);
bool operator!= (const Matrix& x, const Matrix& y);
double drand(const double& begin, const double& end);

#endif