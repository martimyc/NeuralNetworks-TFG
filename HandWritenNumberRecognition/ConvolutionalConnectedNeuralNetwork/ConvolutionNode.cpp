#include "ConvolutionNode.h"

#include <iostream>

ConvolutionNode::ConvolutionNode(std::vector<ConvolutionNode::Kernel>& kernels): ConvolutionLayerNode(), ComputationNode(NT_CONVOLUTION), kernels(kernels)
{}

ConvolutionNode::ConvolutionNode(ConvolutionNode::Kernel & kernel): ConvolutionLayerNode(), ComputationNode(NT_CONVOLUTION), kernels()
{
	kernels.push_back(kernel);
}

ConvolutionNode::~ConvolutionNode()
{}

void ConvolutionNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
{
	if (kernels.size() == 0)
	{
		std::cerr << "ConvolutionNode - Forward - No kernels" << std::endl;
		return;
	}

	output.reserve(kernels.size() * inputs.size());

	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		for (std::vector<Kernel>::const_iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++)
		{
			int rows = input->rows() - kernel->Rows() + 1;
			int cols = input->cols() - kernel->Cols() + 1;

			output.push_back(Eigen::MatrixXd(rows, cols));
			Eigen::MatrixXd& mat = output.back();

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					mat(i, j) = ComputeForwardPixel(i, j, *kernel, *input);
				}
			}
		}
	}
}

void ConvolutionNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
{
	output.reserve(inputs.size());

	{
		std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();

		for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++, gradient++)
		{
			for (std::vector<Kernel>::const_iterator kernel = kernels.begin(); kernel != kernels.end(); kernel++, gradient++)
			{
				int rows = gradient->rows();
				int cols = gradient->cols();

				output.push_back(Eigen::MatrixXd(rows, cols));
				Eigen::MatrixXd& mat = output.back();

				for (int i = 0; i < rows; i++)
				{
					for (int j = 0; j < cols; j++)
					{
						ComputeBackwardPixel(i, j, *gradient, *input, *kernel, mat);
					}
				}
			}
		}
	}
}

void ConvolutionNode::UpdateWeightsAndBiases(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float training_rate)
{
	// CHEK
	{
		std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();

		for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++, gradient++)
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
}

void ConvolutionNode::UpdateWeightsAndBiasesRegular(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float eta, int mini_batch_size, float lambda)
{
	// CHEK
	{
		std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();

		for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++, gradient++)
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
			// Maybe no input cus derivative of Ax = A?
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
			output(i,j) = gradient(x, y) * kernel.Weights()(i, j) * input(x + i, y + j);
		}
	}
}