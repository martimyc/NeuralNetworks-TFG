#include "ConvolutionLayer.h"

#include <iostream>

// Nodes
#include "ConvolutionLayerNode.h"
#include "ConvolutionNode.h"

ConvolutionLayer::ConvolutionLayer(bool regularization): Layer(LT_CONVOLUTION, regularization)
{}

ConvolutionLayer::~ConvolutionLayer()
{
	for (std::vector<ConvolutionLayerNode*>::iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		delete *node;
	}
}

const Eigen::MatrixXd ConvolutionLayer::FeedForward(const Eigen::MatrixXd& input)
{
	if (nodes.size() == 0)
	{
		std::cerr << "ConvolutionLayer - FeedForward - No nodes in this layer" << std::endl;
		return Eigen::MatrixXd();
	}

	if (weights_node == nullptr)
	{
		std::cerr << "ConvolutionLayer - FeedForward - No weights node" << std::endl;
		return Eigen::MatrixXd();
	}

	std::vector<Eigen::MatrixXd> input_vec;

	MatToVec(input, input_vec);

	std::vector<Eigen::MatrixXd> output_vec;

	for (std::vector<ConvolutionLayerNode*>::const_iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		(*node)->Forward(input_vec, output_vec);
		inputs.push_back(output_vec);
		input_vec = output_vec;
	}

	Eigen::MatrixXd output(output_vec.size(), 1);

	VecToMat(output_vec, output);

	return output;
}

const Eigen::MatrixXd ConvolutionLayer::BackPropagate(const Eigen::MatrixXd & gradient, float eta, float mini_batch_size, float lambda) const
{
	std::vector<Eigen::MatrixXd> gradient_vec(gradient.size());

	MatToVec(gradient, gradient_vec);

	std::vector<Eigen::MatrixXd> weights_gradient;
	std::vector<Eigen::MatrixXd> weights_input;

	std::vector<Eigen::MatrixXd> output_vec;

	{
		std::vector<ConvolutionLayerNode*>::const_reverse_iterator node = nodes.rbegin();
		std::vector<std::vector<Eigen::MatrixXd>>::const_reverse_iterator input = inputs.rbegin();

		for (; node != nodes.rend(); node++, input++)
		{
			(*node)->Backward(*input, gradient_vec, output_vec);

			if (*node == weights_node)
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

Eigen::VectorXd ConvolutionLayer::RandomInitBias(int num_neurons)
{
	std::normal_distribution<double> distribution(0.0, 1.0);

	Eigen::VectorXd new_vec(num_neurons);

	for (int i = 0; i < num_neurons; i++)
	{
		new_vec(i) = distribution(engine);
	}
	return new_vec;
}

Eigen::MatrixXd ConvolutionLayer::RandomInitWeight(int num_neurons, int previous_layer_neurons)
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