#include "makural.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layersSizes) : layers_(layersSizes.size() - 1) {
	for (int i = 0; i < layers_.size(); ++i) {
		layers_[i] = new Layer(layersSizes[i], layersSizes[i + 1]);
	}
}
const Matrix& NeuralNetwork::calculateOutputs(const Matrix& input) {
	layers_[0]->calculateOutput(input);
	for (int i = 1; i < layers_.size(); ++i) {
		layers_[i]->calculateOutput(layers_[i - 1]->getBeforeActivation());
	}
}