#include "FullyConnectedNode.h"

#include "imgui.h"

FullyConnectedNode::FullyConnectedNode(std::mt19937 & engine, int num_neurons, int num_connections):
	ComputationNode(NT_FULLY_CONNECTED),
	weights(num_neurons, num_connections),
	biases(num_neurons, 1)
{
	// Weigths
	std::normal_distribution<double> distribution_weights(0.0, 1.0 / sqrt(num_connections));

	for (int i = 0; i < num_neurons; i++)
	{
		for (int j = 0; j < num_connections; j++)
		{
			weights(i, j) = distribution_weights(engine);
		}
	}

	// Biases
	std::normal_distribution<double> distribution_biases(0.0, 1.0);

	for (int i = 0; i < biases.rows(); i++)
	{
		biases(i, 0) = distribution_biases(engine);
	}

	// Debug
	/*for (int i = 0; i < num_neurons; i++)
	{
		for (int j = 0; j < num_connections; j++)
		{
			weights(i, j) = 0.5;
		}
	}

	for (int i = 0; i < biases.rows(); i++)
	{
		biases(i, 0) = 0.5;
	}*/
}

FullyConnectedNode::FullyConnectedNode(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& biases):
	ComputationNode(NT_FULLY_CONNECTED),
	weights(weights),
	biases(biases)
{}

FullyConnectedNode::~FullyConnectedNode()
{}

void FullyConnectedNode::Forward(Eigen::VectorXd & input) const
{
	input = weights * input + biases;
}

void FullyConnectedNode::Backward(const Eigen::VectorXd & input, Eigen::VectorXd & gradient)
{
	gradient = weights.transpose() * gradient;
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
	weights -= (1 - eta * (lambda / weights.size())) * (eta / mini_batch_size) * (gradient * input.transpose());
}

bool FullyConnectedNode::UINode() const
{
	return ImGui::Button("Fully\nConnected\nNode", BUTTON_SIZE);
}

void FullyConnectedNode::UIDescription() const
{
	ImGui::TextWrapped("Fully connected nodes provide an array of neurons all connected to each input.");
}