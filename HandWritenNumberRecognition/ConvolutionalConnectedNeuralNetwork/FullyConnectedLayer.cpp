#include "FullyConnectedLayer.h"

#include <math.h>

#include "NeuralNetwork.h"

//Nodes
#include "FullyConnectedNode.h"

FullyConnectedLayer::FullyConnectedLayer(int layer_neurons, int previous_layer_neurons, bool regularization) :
	Layer(LT_FULLY_CONNECTED, regularization),
	num_neurons(layer_neurons),
	num_weights(previous_layer_neurons)
{
	// TODO init nodes
	// Random initialization

}

FullyConnectedLayer::~FullyConnectedLayer()
{
	for (std::vector<FullyConnectedLayerNode*>::iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		delete *node;
	}
}

const Eigen::MatrixXd FullyConnectedLayer::FeedForward(const Eigen::MatrixXd& input)
{
	if (nodes.size() == 0)
	{
		std::cerr << "FullyConnectedLayer - FeedForward - No nodes in this layer" << std::endl;
		return Eigen::MatrixXd();
	}

	if (weights_node == nullptr)
	{
		std::cerr << "FullyConnectedLayer - FeedForward - No weights node" << std::endl;
		return Eigen::MatrixXd();
	}

	Eigen::VectorXd input_vec(input.size());

	MatToVec(input, input_vec);

	Eigen::VectorXd output_vec;

	for (std::vector<FullyConnectedLayerNode*>::const_iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		(*node)->Forward(input_vec, output_vec);
		inputs.push_back(output_vec);
		input_vec = output_vec;
	}

	Eigen::MatrixXd output(output_vec.size(),1);

	VecToMat(output_vec, output);

	return output;
}

const Eigen::MatrixXd FullyConnectedLayer::BackPropagate(const Eigen::MatrixXd & gradient, float eta, float mini_batch_size, float lambda) const
{
	Eigen::VectorXd gradient_vec(gradient.size());

	MatToVec(gradient, gradient_vec);

	Eigen::VectorXd weights_gradient;
	Eigen::VectorXd weights_input;

	Eigen::VectorXd output_vec;

	{
		std::vector<FullyConnectedLayerNode*>::const_reverse_iterator node = nodes.rbegin();
		std::vector<Eigen::VectorXd>::const_reverse_iterator input = inputs.rbegin();

		for (; node != nodes.rend(); node++, input++)
		{
			(*node)->Backward(*input, gradient_vec, output_vec);
			
			if(*node == weights_node)
			{
				weights_gradient = gradient_vec;
				weights_input = *input;
			}

			gradient_vec = output_vec;
		}
	}

	if (!regularization)
	{
		weights_node->UpdateWeightsAndBiases(weights_gradient, weights_input, eta / mini_batch_size);
	}
	else
	{
		weights_node->UpdateWeightsAndBiasesRegular(weights_gradient, weights_input, eta, mini_batch_size, lambda);
	}

	Eigen::MatrixXd output(output_vec.size(), 1);

	VecToMat(output_vec, output);

	return output;
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
