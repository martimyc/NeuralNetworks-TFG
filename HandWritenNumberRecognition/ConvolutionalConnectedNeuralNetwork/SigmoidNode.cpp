#include "SigmoidNode.h"

SigmoidNode::SigmoidNode(): ComputationNode(NT_SIGMOID)
{}

SigmoidNode::~SigmoidNode()
{}

void SigmoidNode::Forward(const Eigen::VectorXd & input, Eigen::VectorXd & output) const
{
	output = input.unaryExpr(&Sigmoid);
}

void SigmoidNode::Backward(const Eigen::VectorXd & input, const Eigen::VectorXd & gradient, Eigen::VectorXd & output)
{
	output = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void SigmoidNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
{
	output.reserve(inputs.size());
	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		output.push_back(input->unaryExpr(&Sigmoid));
	}
}

void SigmoidNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
{
	output.reserve(inputs.size());
	{
		std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();
		std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();
		for (; input != inputs.end(); input++, gradient++)
		{
			output.push_back(input->unaryExpr(&Derivative).asDiagonal() * (*gradient));
		}
	}
}

double SigmoidNode::Sigmoid(double input)
{
	return 1.0 / (1.0 + exp(input));
}

double SigmoidNode::Derivative(double input)
{
	return Sigmoid(input)*(1.0 - Sigmoid(input));
}
