#include "ConvolutionLayer.h"

#include <iostream>
#include <math.h>

#include "imgui.h"

#include "Globals.h"

// Nodes
#include "ConvolutionLayerNode.h"
#include "ConvolutionNode.h"
#include "L2PoolingNode.h"
#include "MaxPoolingNode.h"
#include "ReluNode.h"
#include "SigmoidNode.h"
#include "TanhNode.h"

ConvolutionLayer::ConvolutionLayer(int k_size, POOLING pooling, ACTIVATION_FUNCTION activation_function, int num_filters, int input_image_size, bool regularization):
	Layer(LT_CONVOLUTION, regularization),
	k_size(k_size),
	pooling(pooling),
	num_filters(num_filters),
	input_image_h(sqrt(input_image_size)),
	input_image_w(input_image_h),
	focused(nullptr)
{
	// Output sizes
	output_image_h = (int)ceil(((float)input_image_h - (float)k_size + 1.0f) / (float)sqrt(POOLING_WINDOW));
	output_image_w = output_image_h;

	AddConvolutionNode(k_size, num_filters);

	switch (activation_function)
	{
	case AF_SIGMOID: AddSigmoidNode(); break;
	case AF_RELU: AddReluNode(); break;
	case AF_TANH: AddTanhNode(); break;
	default:
		std::cerr << "FullyConnectedLayer - Constructor - Unidentified activation function" << std::endl;
		break;
	}

	switch (pooling)
	{
	case P_L2: AddL2PoolingNode(); break;
	case P_MAX: AddMaxPoolingNode(); break;

	default:
		std::cerr << "ConvolutionLayer - ConvolutionLayer - Unidentified pooling" << std::endl;
		break;
	}

	// UI
	focused = nodes.front();
}

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

	if (convolution_node == nullptr)
	{
		std::cerr << "ConvolutionLayer - FeedForward - No convolution node" << std::endl;
		return Eigen::MatrixXd();
	}

	std::vector<Eigen::MatrixXd> input_vec;

	MatToVecFw(input, input_vec);

	for (std::vector<ConvolutionLayerNode*>::const_iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		// Save input
		inputs.push_back(input_vec);

		// Compute
		(*node)->Forward(input_vec);
	}

	Eigen::MatrixXd output(input_vec.size(), 1);

	VecToMat(input_vec, output);

	return output;
}

const Eigen::MatrixXd ConvolutionLayer::BackPropagate(const Eigen::MatrixXd & gradient, float eta, float mini_batch_size, float lambda) const
{
	std::vector<Eigen::MatrixXd> gradient_vec;

	MatToVecBack(gradient, gradient_vec);

	std::vector<Eigen::MatrixXd> weights_gradient;

	std::vector<ConvolutionLayerNode*>::const_reverse_iterator node = nodes.rbegin();
	std::vector<std::vector<Eigen::MatrixXd>>::const_reverse_iterator input = inputs.rbegin();

	for (; node != nodes.rend(); node++, input++)
	{
		// Save to update weights
		if (*node == convolution_node)
		{
			weights_gradient = gradient_vec;
		}

		// Compute
		(*node)->Backward(*input, gradient_vec);
	}

	if (!regularization)
	{
		convolution_node->UpdateWeightsAndBiases(weights_gradient, inputs.front(), eta / mini_batch_size);
	}
	else
	{
		convolution_node->UpdateWeightsAndBiasesRegular(weights_gradient, inputs.front(), eta, mini_batch_size, lambda);
	}

	Eigen::MatrixXd output(gradient_vec.size(), 1);

	VecToMat(gradient_vec, output);

	return output;
}

void ConvolutionLayer::CleanUp()
{
	inputs.clear();
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

void ConvolutionLayer::UI()
{
	ImGui::TextWrapped("Convolution Layer");

	for (std::vector<ConvolutionLayerNode*>::const_iterator it = nodes.begin(); it != nodes.end(); it++)
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

void ConvolutionLayer::AddConvolutionNode(int k_size, int num_filters)
{
	convolution_node = new ConvolutionNode(engine, k_size, num_filters);
	nodes.push_back(convolution_node);
}

void ConvolutionLayer::AddL2PoolingNode()
{
	nodes.push_back(new L2PoolingNode());
}

void ConvolutionLayer::AddMaxPoolingNode()
{
	nodes.push_back(new MaxPoolingNode());
}

void ConvolutionLayer::AddReluNode()
{
	nodes.push_back(new ReluNode());
}

void ConvolutionLayer::AddSigmoidNode()
{
	nodes.push_back(new SigmoidNode());
}

void ConvolutionLayer::AddTanhNode()
{
	nodes.push_back(new TanhNode());
}

void ConvolutionLayer::Debug()
{
	Eigen::MatrixXd mat(28,28);
	mat <<
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

	Eigen::IOFormat fmt;

	std::vector<Eigen::MatrixXd> input_vec;

	MatToVecFw(mat, input_vec);

	for (std::vector<ConvolutionLayerNode*>::const_iterator node = nodes.begin(); node != nodes.end(); node++)
	{
		// Save input
		inputs.push_back(input_vec);

		std::cerr << "Input:\n" << input_vec.front().format(fmt) << std::endl;

		// Compute
		(*node)->Forward(input_vec);
	}

	Eigen::MatrixXd output(input_vec.size(), 1);

	std::cerr << "Output:\n" << input_vec.front().format(fmt) << std::endl;

	VecToMat(input_vec, output);

	std::cerr << "Output after vecToMat:\n" << output.format(fmt) << std::endl;
}

void ConvolutionLayer::MatToVecFw(const Eigen::MatrixXd & input, std::vector<Eigen::MatrixXd>& output) const
{
	int num_images = input.size() / (input_image_h * input_image_w);

	output.reserve(num_images);

	for (int i = 0; i < num_images; i++)
	{
		output.push_back(input.block(i * input_image_w, 0, input_image_w, input_image_h));
	}
}

void ConvolutionLayer::MatToVecBack(const Eigen::MatrixXd & input, std::vector<Eigen::MatrixXd>& output) const
{
	int image_size = (output_image_h * output_image_w);
	int num_images = input.size() / image_size;

	output.reserve(num_images);

	for (int i = 0; i < num_images; i++)
	{
		/*
		We check cols() because images and neurons are stacked row wise
		ex. for 5x5 images and 10 images the input would have 50 rows and 5 cols
		*/

		if (input.cols() == 1)
		{
			output.push_back(Eigen::MatrixXd(output_image_w, output_image_h));
			Eigen::MatrixXd& image = output.back();

			for (int k = 0; k < output_image_h; k++) // Rows
			{
				for (int j = 0; j < output_image_w; j++) // Cols
				{
					image(k, j) = input(i * image_size + k * output_image_h + j, 0);
				}
			}
		}
		else if(input.cols() == output_image_h)
		{
			output.push_back(input.block(i * output_image_w, 0, output_image_w, output_image_h));
		}
		else
		{
			std::cerr << "ConvolutionLayer - MatToVecback - Bad matrix size" << std::endl;
			return;
		}
	}
}

void ConvolutionLayer::VecToMat(const std::vector<Eigen::MatrixXd>& input, Eigen::MatrixXd & output) const
{
	int image_cols = input.front().cols();
	int image_rows = input.front().rows();

	output = Eigen::MatrixXd(image_rows * input.size(), image_cols);

	for (int i = 0; i < input.size(); i++)
	{
		output.block(i * image_rows, 0, image_rows, image_cols) = input[i];
	}
}
