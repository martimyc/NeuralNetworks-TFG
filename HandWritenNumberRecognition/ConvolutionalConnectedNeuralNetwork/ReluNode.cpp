#include "ReluNode.h"

ReluNode::ReluNode(): ComputationNode(NT_RELU)
{}

ReluNode::~ReluNode()
{}

void ReluNode::Forward(const Eigen::VectorXd & input, Eigen::VectorXd & output) const
{
	output = input.unaryExpr(&Relu);
}

void ReluNode::Backward(const Eigen::VectorXd & input, const Eigen::VectorXd & gradient, Eigen::VectorXd & output)
{
	output = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void ReluNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
{
	output.reserve(inputs.size());
	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		output.push_back(input->unaryExpr(&Relu));
	}
}

void ReluNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
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

double ReluNode::Relu(double input)
{
	if (input <= 0)
	{
		return 0.0;
	}
	else
	{
		return input;
	}
}

double ReluNode::Derivative(double input)
{
	if (input <= 0)
	{
		return 0.0;
	}
	else
	{
		return input;
	}
}
