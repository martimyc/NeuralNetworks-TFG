#ifndef LAYER
#define LAYER

#include <vector>
#include <random>

#include "Eigen/Dense"

enum COST_FUNCTION;

enum ACTIVATION_FUNCTION
{
	AF_SIGMOID

};

enum LAYER_TYPE
{
	LT_FULLY_CONNECTED,
	LT_CONVOLUTION
};

class FullyConnectedLayer;

class Layer
{
public:
	Layer(LAYER_TYPE);
	~Layer();

	// Work
	virtual const Eigen::VectorXd& FeedForward(const Eigen::MatrixXd& input) = 0;
	virtual const Eigen::VectorXd BackPropagate(const Eigen::MatrixXd& next_error, const Eigen::MatrixXd& next_weights) const = 0;
	virtual void UpdateWeightsAndBiases(const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous_activation, float training_rate) = 0;
	virtual void UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous_activation, float eta, int mini_batch_size, float lambda) = 0;

	// Activation Functions
	Eigen::VectorXd ActivationFunction(const Eigen::VectorXd& vec) const;
	Eigen::MatrixXd ActivationFunction(const Eigen::MatrixXd& mat) const;
	static double Sigmoid(double z);
	static double SigmoidPrime(double z);

	// Polimorphism
	FullyConnectedLayer* AsFullyConnected();

	// Getters
	virtual inline const Eigen::VectorXd& GetZ() = 0;
	virtual inline const Eigen::VectorXd& GetActivation() = 0;
	virtual inline const Eigen::MatrixXd& GetWeights() = 0;
	virtual inline const Eigen::VectorXd& GetBiases() = 0;
	virtual inline const int GetNumNeurons() const = 0;

protected:
	LAYER_TYPE type;
	ACTIVATION_FUNCTION activation_function;

	// Random
	static std::random_device rand_device;
	static std::mt19937 engine;
};

#endif //!LAYER
