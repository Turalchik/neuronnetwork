#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include "layer.h"
#include"costfunctions.h"
#include<string>

class NeuralNetwork {
	std::vector<Layer*> layers_;
	ActivationFunction* general_activation_func_;
	ActivationFunction* output_activation_func_;
	CostFunction* cost_func_;

	void changeWeights(const Matrix& convergence_step);
	void applyGeneralBackpropagationAlgorithm(const Matrix& ourOutputs, const Matrix& actualOutputs);
	void applySuitBackpropagationAlgorithm(const Matrix& ourOutputs, const Matrix& actualOutputs);
	void optimizerSGD(const std::vector<Matrix*>& data_train, size_t batchBegins, size_t batchEnds, 
					  const std::vector<Matrix*>& answers, const double& learning_rate);

	static void shuffle(std::vector<Matrix*>& data, std::vector<Matrix*>& answers);

public:
	NeuralNetwork(const std::vector<int>& layers_sizes, ActivationFunction* general_activation_function, 
				  ActivationFunction* output_activation_function, CostFunction* cost_func);

	NeuralNetwork(const char* input_filename, ActivationFunction* general_activation_function,
		ActivationFunction* output_activation_function, CostFunction* cost_func);

	void train(std::vector<Matrix*>& data_train, std::vector<Matrix*>& answers_train,
		std::vector<Matrix*>& data_test, std::vector<Matrix*>& answers_test, const size_t& epochs, const size_t& butchSize);

	Matrix calculateAnswer(const Matrix& input);

	void save(const std::string& output_filename) const;
	int predict(const Matrix& input);
	double calculateAccuracy(const std::vector<Matrix*>& data_test, const std::vector<Matrix*> answers_test);
	double calculateAverageLoss(const std::vector<Matrix*>& data_test, const std::vector<Matrix*> answers_test);

	~NeuralNetwork();
};

#endif
