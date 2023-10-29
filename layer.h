#ifndef LAYER_MAKURAL
#define LAYER_MAKURAL

#include "matrixlab.h"
#include "activationFunctions.h"

class Layer {
protected:
	Matrix before_activation_;
	Matrix weights_;
	Matrix after_activation_;
	Matrix biases_;

	Matrix gradient_nodes_weights_;
	Matrix gradient_nodes_biases_;
public:
	Layer(int input_size, int output_size);
	virtual const Matrix& calculateLayerOutput(const Matrix& input_data, const ActivationFunction* activationFunc) = 0;
	const Matrix& getBeforeActivation() const;
	const Matrix& getAfterActivation() const;
	const Matrix& getWeights() const;
	const Matrix& getBiases() const;
	const Matrix& getGradientsForWeights() const;
	const Matrix& getGradientsForBiases() const;
	void averageGradient(const Matrix& butch_size);
	void putGradientIntoCurrentLayer(Matrix&& weights, Matrix&& biases);
	void addGradientToCurrentLayer(const Matrix& weights, const Matrix& biases);
	void changeWeightsAndBiasesByGradient(const Matrix& convergence_step);
	void initializeWeightsAndBiases();
	void loadWeightsAndBiases(std::ifstream& inFile);
	std::ofstream& save(std::ofstream& outFile) const;
};

class Input : public Layer {
public:
	Input(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateLayerOutput(const Matrix& input, const ActivationFunction* activationFunc);
};

class Dense : public Layer {
public:
	Dense(int input_size, int output_size) : Layer(input_size, output_size){}
	const Matrix& calculateLayerOutput(const Matrix& input, const ActivationFunction* activationFunc);
};

#endif
