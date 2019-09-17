#include "FullyConnectedNode.h"

FullyConnectedNode::FullyConnectedNode(int* input, int* output):
	ComputationalNode(NT_FULLY_CONNECTED),
	input_neurons(),
	output_neurons()
{
	for (int i = 0; i < 3; i++)
	{
		input_neurons[i] = input[i];
		output_neurons[i] = output[i];
	}
}

FullyConnectedNode::~FullyConnectedNode()
{}

const Eigen::MatrixXd FullyConnectedNode::Forward(const Eigen::MatrixXd & input)
{
	return weights * input + biases;
}

Eigen::MatrixXd FullyConnectedNode::Backward(const Eigen::MatrixXd & input, const Eigen::MatrixXd & derivative)
{
	// TODO test cus chain rule sais one and expirience sais the other
	//return input * weights.transpose() * derivative;
	return weights.transpose() * derivative;
}

void FullyConnectedNode::UpdateWeightsAndBiases(const Eigen::MatrixXd & gradient, const Eigen::MatrixXd & input, float training_rate)
{
	/*
	To modify biases we just multiply each bias by its error since the error is equal to the cost derivative in regards to bias.
	For weights it is similar but the activation value of the previous layer must also be taken into acount since the output error of
	each weight depends on the activation of the neuron it is connected to
	*/

	biases -= training_rate * gradient;
	weights -= training_rate * (gradient * input.transpose());
}

void FullyConnectedNode::UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd & gradient, const Eigen::MatrixXd & input, float eta, int mini_batch_size, float lambda)
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

	biases -= (eta / mini_batch_size) * gradient;
	weights -= (1 - eta * (lambda / weights.rows())) * (eta / mini_batch_size) * (gradient * input.transpose());
}