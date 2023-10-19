#ifndef LAYER_MAKURAL
#define LAYER_MAKURAL

#include "matrixlab.h"

class Layer {
	Matrix beforeActivation_;
	Matrix weights_;
	Matrix afterActivation_;
	Matrix biases_;

	Matrix gradientNodesWeights_;
	Matrix gradientNodesBiases_;
public:
	Layer(int, int);
	const Matrix& calculateOutput(const Matrix&);

	const Matrix& getBeforeActivation() {
		return beforeActivation_;
	}
};


#endif
