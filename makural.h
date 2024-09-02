#ifndef MODEL_MAKURAL
#define MODEL_MAKURAL

#include <vector>
#include <string>
#include "layer.h"
#include "costfunctions.h"

class NeuralNetwork;

typedef std::vector<Eigen::MatrixXd*> dataVec;

typedef void(NeuralNetwork::* ptrFun)(const dataVec& data_train, size_t batchBegins, size_t batchEnds,
									  const dataVec& answers, const double& learning_rate);

class NeuralNetwork {
	std::vector<Layer*> layers_;
	HiddenActivationFunction* general_activation_func_;
	OutputActivationFunction* output_activation_func_;
	CostFunction* cost_func_;

	void changeWeights(const double& convergence_step);
	void applyGeneralBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void applySuitBackpropagationAlgorithm(const Eigen::MatrixXd& ourOutputs, const Eigen::MatrixXd& actualOutputs);
	void optimizerSGD(const dataVec& data_train, size_t batchBegins, size_t batchEnds,
					  const dataVec& answers, const double& learning_rate);
	void Adam(const dataVec& data_train, size_t batchBegins, size_t batchEnds,
		const dataVec& answers, const double& learning_rate);

	static void shuffle(dataVec& data, dataVec& answers, const size_t& from, const size_t& to);
	ptrFun getOptimizer(const char* optimizer_name);

public:

	NeuralNetwork(const std::vector<int>& layers_sizes, const char* general_activation_function, 
				  const char* output_activation_function, const char* cost_func);

	NeuralNetwork(const char* input_filename);
	void train(dataVec& data_train, dataVec& answers_train,
		const double& validation_split, const char* optimizer_name, const size_t& epochs, const size_t& butchSize);

	Eigen::MatrixXd answerVec(const Eigen::MatrixXd& input);
	void save(const std::string& output_filename) const;
	int predict(const Eigen::MatrixXd& input);
	double accuracy(const dataVec& data_test, const dataVec& answers_test);
	double averageLoss(const dataVec& data_test, const dataVec& answers_test, 
								const size_t& from, const size_t& to);

	void reset();
	~NeuralNetwork();
};


#endif
