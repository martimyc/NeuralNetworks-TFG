#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <random>

#include "Eigen/Dense"

// Layers
class Layer;
class FullyConnectedLayer;
class ConvolutionLayer;

class MNIST;

enum STATE {
	S_READY,
	S_TRAINING,
	S_DONE
};

enum COST_FUNCTION {
	CF_QUADRATIC = 0,
	CF_CROSS_ENTHROPY
};

class NeuralNetwork
{
public:
	// Constructor & Destructor
	NeuralNetwork(COST_FUNCTION cost_function = CF_QUADRATIC, bool regularization = false);
	virtual ~NeuralNetwork();

	// Work
	const Eigen::MatrixXd& FeedForward(const Eigen::MatrixXd& input) const;
	void BackPropagation(const Eigen::MatrixXd& cost, const std::vector<Eigen::MatrixXd> & inputs_vec, float eta, int mini_batch_size, float lambda);
	void SGD(const std::vector<MNIST*>& training_data, int epochs, int mini_batch_size, float eta, float lambda = 0.0000f);
	int ComputeResult(Eigen::MatrixXd& input) const;

	// Layers
	FullyConnectedLayer * AddFullyConnectedLayer(int layer_neurons, int previous_layer_neurons, bool regularization);
	ConvolutionLayer * AddConvolutionLayer(bool regularization);

	// Getters
	inline const STATE& GetState() const { return state; }

	// UI
	void Info() const;

private:
	// Work
	void UpdateWithMiniBatch(std::vector<MNIST*>& mini_batch, float eta, float lambda);

	// Cost functions
	Eigen::MatrixXd Delta(const Eigen::MatrixXd& activation, const Eigen::MatrixXd& desired); // Computes overall cost based on the diferent cost functions

	// Test
	void TestOnValidation();
	void TestOnTest();
	void TestOnTraining();

	// Getters
	int GetResult(const Eigen::MatrixXd& output) const;

	// Analytics
	float Cost(const Eigen::MatrixXd& output, const Eigen::MatrixXd& desired);

private:
	std::vector<Layer*> layers;
	STATE state;
	COST_FUNCTION cost_function;
	bool regularization;
};

#endif //!NEURALNETWORK