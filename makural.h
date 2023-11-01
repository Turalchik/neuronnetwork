#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include<string>
#include "layer.h"
#include"costfunctions.h"
#include<time.h>

class NeuralNetwork;

typedef void(NeuralNetwork::* ptrFun)(const std::vector<Eigen::MatrixXd*>& data_train, size_t batchBegins, size_t batchEnds,
									  const std::vector<Eigen::MatrixXd*>& answers, const double& learning_rate);

class NeuralNetwork {
	std::vector<Layer*> layers_;
	HiddenActivationFunction* general_activation_func_;
	OutputActivationFunction* output_activation_func_;
	CostFunction* cost_func_;

	void changeWeights(const double& convergence_step);
	void applyGeneralBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void applySuitBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void optimizerSGD(const std::vector<Eigen::MatrixXd*>& data_train, size_t batchBegins, size_t batchEnds,
					  const std::vector<Eigen::MatrixXd*>& answers, const double& learning_rate);

	static void shuffle(std::vector<Eigen::MatrixXd*>& data, std::vector<Eigen::MatrixXd*>& answers, const size_t& from, const size_t& to);
	ptrFun getOptimizer(const char* optimizer_name);

public:

	NeuralNetwork(const std::vector<int>& layers_sizes, const char* general_activation_function, 
				  const char* output_activation_function, const char* cost_func);

	NeuralNetwork(const char* input_filename);
	void train(std::vector<Eigen::MatrixXd*>& data_train, std::vector<Eigen::MatrixXd*>& answers_train,
		const double& validation_split, const char* optimizer_name, const size_t& epochs, const size_t& butchSize);

	Eigen::MatrixXd answerVec(const Eigen::MatrixXd& input);
	void save(const std::string& output_filename) const;
	int predict(const Eigen::MatrixXd& input);
	double accuracy(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test);
	double averageLoss(const std::vector<Eigen::MatrixXd*>& data_test, const std::vector<Eigen::MatrixXd*> answers_test, 
								const size_t& from, const size_t& to);

	void reset();
	~NeuralNetwork();
};


#endif
