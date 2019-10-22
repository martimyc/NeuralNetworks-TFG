#include "FullyConnectedLayer.h"

#include <math.h>
#include <iostream>

#include "imgui.h"

#include "NeuralNetwork.h"

//Nodes
#include "FullyConnectedNode.h"
#include "ReluNode.h"
#include "SigmoidNode.h"
#include "TanhNode.h"
#include "SoftMaxNode.h"

FullyConnectedLayer::FullyConnectedLayer(int layer_neurons, int previous_layer_neurons, ACTIVATION_FUNCTION activation_funct, bool regularization) :
	Layer(LT_FULLY_CONNECTED, regularization),
	num_neurons(layer_neurons),
	num_weights(previous_layer_neurons),
	focused(nullptr)
{
	AddFullyConnectedNode(layer_neurons, previous_layer_neurons);

	switch (activation_funct)
	{
	case AF_SIGMOID: AddSigmoidNode(); break;
	case AF_RELU: AddReluNode(); break;
	case AF_TANH: AddTanhNode(); break;
	case AF_SOFTMAX: AddSoftMaxNode(); break;
	default:
		std::cerr << "FullyConnectedLayer - Constructor - Unidentified activation function" << std::endl;
		break;
	}

	focused = nodes.front();
}

FullyConnectedLayer::~FullyConnectedLayer()
{
	for (std::vector<FullyConnectedLayerNode*>::reverse_iterator node = nodes.rbegin(); node != nodes.rend(); node++)
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

	Eigen::VectorXd input_vec(input.rows() * input.cols());

	MatToVec(input, input_vec);

	for (std::vector<FullyConnectedLayerNode*>::const_iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		// Save Input
		inputs.push_back(input_vec);

		// Compute and update vec
		(*node)->Forward(input_vec);
	}

	Eigen::MatrixXd output(input_vec.size(),1);

	VecToMat(input_vec, output);

	return output;
}

const Eigen::MatrixXd FullyConnectedLayer::BackPropagate(const Eigen::MatrixXd & gradient, float eta, float mini_batch_size, float lambda) const
{
	Eigen::VectorXd gradient_vec(gradient.size());

	MatToVec(gradient, gradient_vec);

	// To update weights
	Eigen::VectorXd update_vec(gradient.size());

	// Back prop
	{
		std::vector<FullyConnectedLayerNode*>::const_reverse_iterator node = nodes.rbegin();
		std::vector<Eigen::VectorXd>::const_reverse_iterator input = inputs.rbegin();

		for (; node != nodes.rend(); node++, input++)
		{
			// Save gradient at weights to update
			if (*node == weights_node)
			{
				update_vec = gradient_vec;
			}

			// Compute
			(*node)->Backward(*input, gradient_vec);
		}
	}

	// Update
	if (!regularization)
	{
		// We use the gradient at the node (total gradient by activation funct derivative of Z) & input of the whole layer (previous layer activations) to update the wheights and biases
		weights_node->UpdateWeightsAndBiases(update_vec, inputs.front(), eta / mini_batch_size);
	}
	else
	{
		// We use the gradient at the node (total gradient by activation funct derivative of Z) & input of the whole layer (previous layer activations) to update the wheights and biases
		weights_node->UpdateWeightsAndBiasesRegular(update_vec, inputs.front(), eta, mini_batch_size, lambda);
	}

	Eigen::MatrixXd output(gradient_vec.size(), 1);

	VecToMat(gradient_vec, output);

	return output;
}

void FullyConnectedLayer::CleanUp()
{
	inputs.clear();
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

void FullyConnectedLayer::UI()
{
	ImGui::TextWrapped("Fully Connected Layer");

	for (std::vector<FullyConnectedLayerNode*>::const_iterator it = nodes.begin(); it != nodes.end(); it++)
	{
		if ((*it)->UINode())
		{
			focused = *it;
		}

		if (*it != nodes.back())
		{
			ImGui::SameLine();
		}
	}

	focused->UIDescription();
}

void FullyConnectedLayer::AddFullyConnectedNode(int num_neurons, int num_conections)
{
	weights_node = new FullyConnectedNode(engine, num_neurons, num_conections);
	nodes.push_back(weights_node);
}

void FullyConnectedLayer::AddReluNode()
{
	nodes.push_back(new ReluNode());
}

void FullyConnectedLayer::AddSigmoidNode()
{
	nodes.push_back(new SigmoidNode());
}

void FullyConnectedLayer::AddTanhNode()
{
	nodes.push_back(new TanhNode());
}

void FullyConnectedLayer::AddSoftMaxNode()
{
	nodes.push_back(new SoftMaxNode());
}

void FullyConnectedLayer::Debug()
{
	Eigen::MatrixXd debug_image(4, 4);
	debug_image <<
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0;

	Eigen::IOFormat fmt;
	std::cerr << "Mat:\n" << debug_image.format(fmt) << std::endl;

	Eigen::VectorXd vec(debug_image.size());
	MatToVec(debug_image, vec);
	std::cerr << "Vec:\n" << vec.format(fmt) << std::endl;

	Eigen::MatrixXd mat(vec.rows(), 1);
	VecToMat(vec, mat);
	std::cerr << "Mat:\n" << mat.format(fmt) << std::endl;
}

const Eigen::MatrixXd & FullyConnectedLayer::GetWeights() const
{
	return weights_node->GetWeights();
}

const Eigen::MatrixXd & FullyConnectedLayer::GetBiases() const
{
	return weights_node->GetBiases();
}

void FullyConnectedLayer::MatToVec(const Eigen::MatrixXd & input, Eigen::VectorXd & output) const
{
	//output = Eigen::VectorXd(input);
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			output(i * input.cols() + j) = input(i, j);
		}
	}
}

void FullyConnectedLayer::VecToMat(const Eigen::VectorXd & input, Eigen::MatrixXd & output) const
{
	//output = Eigen::MatrixXd(input);
	for (int i = 0; i < input.size(); i++)
	{
		output(i, 0) = input(i);
	}
}
