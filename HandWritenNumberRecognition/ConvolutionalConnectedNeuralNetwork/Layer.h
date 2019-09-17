#ifndef LAYER
#define LAYER

#include <vector>
#include <iostream>
#include <random>

#include "Eigen/Dense"

enum LAYER_TYPE
{
	LT_FULLY_CONNECTED,
	LT_CONVOLUTION
};

// Layers
class ConvolutionLayer;
class FullyConnectedLayer;

class ComputationNode;

class Layer
{
public:
	Layer(LAYER_TYPE, bool regularization = false);
	~Layer();

	// Work
	virtual const Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input) = 0;
	virtual const Eigen::MatrixXd BackPropagate(const Eigen::MatrixXd& gradient, float eta, float mini_batch_size, float lambda = 0.0f) const = 0;

	// Polimorphism
	FullyConnectedLayer* AsFullyConnected();
	ConvolutionLayer* AsConvolution();

protected:
	LAYER_TYPE type;
	bool regularization;

	// Random
	static std::random_device rand_device;
	static std::mt19937 engine;
};

#endif //!LAYER
