#include "TanhNode.h"

TanhNode::TanhNode(): ComputationNode(NT_TANH)
{}

TanhNode::~TanhNode()
{}

void TanhNode::Forward(const Eigen::VectorXd & input, Eigen::VectorXd & output) const
{
	output = input.unaryExpr(&Tanh);
}

void TanhNode::Backward(const Eigen::VectorXd & input, const Eigen::VectorXd & gradient, Eigen::VectorXd & output)
{
	output = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void TanhNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
{
	output.reserve(inputs.size());
	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		output.push_back(input->unaryExpr(&Tanh));
	}
}

void TanhNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
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

double TanhNode::Tanh(double input)
{
	return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
}

double TanhNode::Derivative(double input)
{
	return 1.0 - Tanh(input) * Tanh(input);
}
