#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include<string>
#include<fstream>
#include<Eigen/Dense>
#include "layer.h"
#include"costfunctions.h"

class NeuralNetwork {
	std::vector<Layer*> layers_;
	ActivationFunction* general_activation_func_;
	ActivationFunction* output_activation_func_;
	CostFunction* cost_func_;

	void changeWeights(const double& convergence_step);
	void applyGeneralBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void applySuitBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void optimizerSGD(const std::vector<Eigen::MatrixXd*>& data_train, size_t batchBegins, size_t batchEnds,
					  const std::vector<Eigen::MatrixXd*>& answers, const double& learning_rate);

	static void shuffle(std::vector<Eigen::MatrixXd*>& data, std::vector<Eigen::MatrixXd*>& answers);

public:
	NeuralNetwork(const std::vector<int>& layers_sizes, ActivationFunction* general_activation_function, 
				  ActivationFunction* output_activation_function, CostFunction* cost_func);

	NeuralNetwork(const char* input_filename, ActivationFunction* general_activation_function,
		ActivationFunction* output_activation_function, CostFunction* cost_func);

	void train(std::vector<Eigen::MatrixXd*>& data_train, std::vector<Eigen::MatrixXd*>& answers_train,
		std::vector<Eigen::MatrixXd*>& data_test, std::vector<Eigen::MatrixXd*>& answers_test, const size_t& epochs, const size_t& butchSize);

	Eigen::MatrixXd calculateAnswer(const Eigen::MatrixXd& input);

	void save(const std::string& output_filename) const;
	int predict(const Eigen::MatrixXd& input);
	double calculateAccuracy(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test);
	double calculateAverageLoss(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test);

	~NeuralNetwork();
};

#endif
