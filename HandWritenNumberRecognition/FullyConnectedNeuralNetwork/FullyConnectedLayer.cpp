#include "FullyConnectedLayer.h"

#include <math.h>

#include "NeuralNetwork.h"

FullyConnectedLayer::FullyConnectedLayer(int layer_neurons, int previous_layer_neurons) :
	Layer(LT_FULLY_CONNECTED),
	weights(layer_neurons, previous_layer_neurons),
	biases(layer_neurons),
	activation_values(Eigen::VectorXd::Zero(layer_neurons)),
	z(Eigen::VectorXd::Zero(layer_neurons)),
	num_neurons(layer_neurons),
	num_weights(previous_layer_neurons)
{
	// Random initialization
	// Bias
	biases = RandomInitBias(layer_neurons);

	// Weights
	weights = RandomInitWeight(layer_neurons, previous_layer_neurons);
}

FullyConnectedLayer::~FullyConnectedLayer()
{}

const Eigen::VectorXd& FullyConnectedLayer::FeedForward(const Eigen::MatrixXd& input)
{
	z = weights * input + biases;
	activation_values = ActivationFunction(z);
	return activation_values;
}

const Eigen::VectorXd FullyConnectedLayer::BackPropagate(const Eigen::MatrixXd& next_error, const Eigen::MatrixXd& next_weights) const
{
	/*
	All layer errors are calculated by multiplying the next layer's weights by the next layer's
	error and that by the variation of the activation function Z in this layer's neurons (sigmoid prime of z)
	To compute this better we create a diagonal matrix with the values resulting from sigmoid prime of Z and multiply it by
	the weights of the previous layer and the prevoius layer error
	*/

	return z.unaryExpr(&Layer::SigmoidPrime).asDiagonal() * next_weights.transpose() * next_error;
}

void FullyConnectedLayer::UpdateWeightsAndBiases(const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous_activation, float training_rate)
{
	/*
	To modify biases we just multiply each bias by its error since the error is equal to the cost derivative in regards to bias.
	For weights it is similar but the activation value of the previous layer must also be taken into acount since the output error of
	each weight depends on the activation of the neuron it is connected to
	*/

	biases -= training_rate * error;
	weights -= training_rate * (error * previous_activation.transpose());
}

void FullyConnectedLayer::UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd & error, const Eigen::MatrixXd & previous_activation, float eta, int mini_batch_size, float lambda)
{
	/*
	To modify biases we just multiply each bias by its error since the error is equal to the cost derivative in regards to bias.
	For weights it is similar but the activation value of the previous layer must also be taken into acount since the output error of
	each weight depends on the activation of the neuron it is connected to
	*/

	/*
	L2 regularization consists in adding one regularization factor to the weight update. This will make the weights change in acordance with other weights
	and will favour small weights
	*/

	biases -= (eta / mini_batch_size) * error;
	weights -= (1 - eta * (lambda / weights.rows())) * (eta / mini_batch_size) * (error * previous_activation.transpose());
}

Eigen::VectorXd FullyConnectedLayer::RandomInitBias(int num_neurons)
{
	std::normal_distribution<double> distribution(0.0, 1.0);

	Eigen::VectorXd new_vec(num_neurons);

	for (int i = 0; i < num_neurons; i++)
	{
		new_vec(i) = distribution(engine);
	}
	return new_vec;
}

Eigen::MatrixXd FullyConnectedLayer::RandomInitWeight(int num_neurons, int previous_layer_neurons)
{
	std::normal_distribution<double> distribution(0.0, 1.0 / sqrt(previous_layer_neurons));

	Eigen::MatrixXd new_mat(num_neurons, previous_layer_neurons);

	for (int i = 0; i < num_neurons; i++)
	{
		for (int j = 0; j < previous_layer_neurons; j++)
		{
			new_mat(i, j) = distribution(engine);
		}
	}

	return new_mat;
}
