#include "ConvolutionNode.h"

#include <iostream>

#include "imgui.h"

ConvolutionNode::ConvolutionNode(std::mt19937 & engine, int k_size, int num_filters): ConvolutionLayerNode(), ComputationNode(NT_CONVOLUTION), kernels(kernels)
{
	for (int i = 0; i < num_filters; i++)
	{
		kernels.push_back(Kernel(k_size));
		Kernel& kernel = kernels.back();
		std::normal_distribution<double> distribution(0.0, 1.0);

		// Weigths
		for (int i = 0; i < k_size; i++)
		{
			for (int j = 0; j < k_size; j++)
			{
				kernel.weights(i, j) = distribution(engine);
			}
		}

		// Bias
		kernel.bias = distribution(engine);

		// Debug
		/*for (int i = 0; i < k_size; i++)
		{
			for (int j = 0; j < k_size; j++)
			{
				kernel.weights(i, j) = 0.5;
			}
		}
		kernel.bias = 0.5;*/
	}
}

ConvolutionNode::~ConvolutionNode()
{}

void ConvolutionNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	if (kernels.size() == 0)
	{
		std::cerr << "ConvolutionNode - Forward - No kernels" << std::endl;
		return;
	}

	std::vector<Eigen::MatrixXd> outputs;

	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++) // Iterate diferent images
	{
		for (std::vector<Kernel>::const_iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++) // Iterate diferent kernels
		{
			int rows = input->rows() - kernel->Rows() + 1;
			int cols = input->cols() - kernel->Cols() + 1;

			outputs.push_back(Eigen::MatrixXd(rows, cols));
			Eigen::MatrixXd& mat = outputs.back();

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					mat(i, j) = ComputeForwardPixel(i, j, *kernel, *input);
				}
			}
		}
	}

	// Fill inputs
	inputs.clear();
	for (std::vector<Eigen::MatrixXd>::const_iterator output = outputs.begin(); output != outputs.end(); output++) // Iterate diferent images
	{
		inputs.push_back(*output);
	}
}

void ConvolutionNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
{
	std::vector<Eigen::MatrixXd> outputs;
	outputs.reserve(inputs.size());

	std::vector<Eigen::MatrixXd>::iterator gradient = gradients.begin();
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

	for (; input != inputs.end(); input++)
	{
		outputs.push_back(Eigen::MatrixXd::Zero(input->rows(), input->cols()));
		Eigen::MatrixXd output = outputs.back();

		for (std::vector<Kernel>::const_iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++, gradient++)
		{
			int rows = gradient->rows();
			int cols = gradient->cols();

			Eigen::MatrixXd mat(input->rows(), input->cols());

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					ComputeBackwardPixel(i, j, *gradient, *input, *kernel, mat);
				}
			}
			output += mat;
		}
	}
	gradients.clear();
	gradients.reserve(outputs.size());

	for (std::vector<Eigen::MatrixXd>::iterator output = outputs.begin(); output != outputs.end(); output++)
	{
		gradients.push_back(*output);
	}
}

void ConvolutionNode::UpdateWeightsAndBiases(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float training_rate)
{
	std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

	for (; input != inputs.end(); input++)
	{
		for (std::vector<Kernel>::iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++, gradient++)
		{
			int rows = gradient->rows();
			int cols = gradient->cols();

			Eigen::MatrixXd weight_gradient(kernel->Rows(), kernel->Cols());
			double bias_gradient = 0.0;

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					ComputeWeightGradient(i, j, *gradient, *input, *kernel, weight_gradient);
					bias_gradient += (*gradient)(i,j);
				}
			}

			kernel->weights -= weight_gradient * training_rate;
			kernel->bias -= bias_gradient * training_rate;
		}
	}
}

void ConvolutionNode::UpdateWeightsAndBiasesRegular(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float eta, int mini_batch_size, float lambda)
{
	// CHEK
	std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

	for (; input != inputs.end(); input++, gradient++)
	{
		for (std::vector<Kernel>::iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++, gradient++)
		{
			int rows = gradient->rows();
			int cols = gradient->cols();

			Eigen::MatrixXd weight_gradient(kernel->Rows(), kernel->Cols());
			double bias_gradient = 0.0;

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					ComputeWeightGradient(i, j, *gradient, *input, *kernel, weight_gradient);
					bias_gradient += (*gradient)(i, j);
				}
			}

			kernel->weights -= (1 - eta * (lambda / kernel->weights.size())) * (eta / mini_batch_size) * weight_gradient;
			kernel->bias -= (eta / mini_batch_size) * bias_gradient;
		}
	}
}

bool ConvolutionNode::UINode() const
{
	return ImGui::Button("Convolution\nNode", BUTTON_SIZE);
}

void ConvolutionNode::UIDescription() const
{
	ImGui::TextWrapped("Convolution nodes apply a wheighted filter on an input matrix.\nThis is similar to how the visual cortex proceses images.\nVery usefull in image recognition.");
}

double ConvolutionNode::ComputeForwardPixel(int x, int y, const Kernel& kernel, const Eigen::MatrixXd & input) const
{
	return kernel.Weights().cwiseProduct(input.block(x, y, kernel.Rows(), kernel.Cols())).sum();
}

void ConvolutionNode::ComputeBackwardPixel(int x, int y, const Eigen::MatrixXd & gradient, const Eigen::MatrixXd & input, const Kernel& kernel, Eigen::MatrixXd & output) const
{
	for (int i = 0; i < kernel.Rows(); i++)
	{
		for (int j = 0; j < kernel.Cols(); j++)
		{
			if (x + i >= input.rows() || y + j >= input.cols())
			{
				continue;
			}
			output(x + i, y + j) += gradient(x, y) * kernel.weights(i, j) * input(x + i, y + j);
		}
	}
}

void ConvolutionNode::ComputeWeightGradient(int x, int y, const Eigen::MatrixXd & gradient, const Eigen::MatrixXd & input, const Kernel & kernel, Eigen::MatrixXd & output)
{
	for (int i = 0; i < kernel.Rows(); i++)
	{
		for (int j = 0; j < kernel.Rows(); j++)
		{
			if (x + i >= input.rows() || y + j >= input.cols())
			{
				continue;
			}
			output(i,j) = gradient(x, y) * kernel.Weights()(i, j) * input(x + i, y + j);
		}
	}
}