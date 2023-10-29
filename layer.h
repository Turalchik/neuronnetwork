#ifndef LAYER_MAKURAL
#define LAYER_MAKURAL

#include "activationFunctions.h"
#include <fstream>

class Layer {
protected:
	Eigen::MatrixXd before_activation_;
	Eigen::MatrixXd weights_;
	Eigen::MatrixXd after_activation_;
	Eigen::MatrixXd biases_;

	Eigen::MatrixXd gradient_nodes_weights_;
	Eigen::MatrixXd gradient_nodes_biases_;
public:
	Layer(int input_size, int output_size);
	virtual const Eigen::MatrixXd& calculateLayerOutput(const Eigen::MatrixXd& input_data, const ActivationFunction* activationFunc) = 0;
	const Eigen::MatrixXd& getBeforeActivation() const;
	const Eigen::MatrixXd& getAfterActivation() const;
	const Eigen::MatrixXd& getWeights() const;
	const Eigen::MatrixXd& getBiases() const;
	const Eigen::MatrixXd& getGradientsForWeights() const;
	const Eigen::MatrixXd& getGradientsForBiases() const;
	void averageGradient(const double& butch_size);
	void putGradientIntoCurrentLayer(Eigen::MatrixXd&& weights, Eigen::MatrixXd&& biases);
	void addGradientToCurrentLayer(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& biases);
	void changeWeightsAndBiasesByGradient(const Eigen::MatrixXd& convergence_step);
	void initializeWeightsAndBiases();
	void loadWeightsAndBiases(std::ifstream& inFile);
	std::ofstream& save(std::ofstream& outFile) const;
};

class Input : public Layer {
public:
	Input(int input_size, int output_size) : Layer(input_size, output_size){}
	const Eigen::MatrixXd& calculateLayerOutput(const Eigen::MatrixXd& input, const ActivationFunction* activationFunc);
};

class Dense : public Layer {
public:
	Dense(int input_size, int output_size) : Layer(input_size, output_size){}
	const Eigen::MatrixXd& calculateLayerOutput(const Eigen::MatrixXd& input, const ActivationFunction* activationFunc);
};

void fillMatrixByRandomNumbers(Eigen::MatrixXd& weights, const double& after_activation_size);
void fillWithZeros(Eigen::MatrixXd& biases);

#endif
