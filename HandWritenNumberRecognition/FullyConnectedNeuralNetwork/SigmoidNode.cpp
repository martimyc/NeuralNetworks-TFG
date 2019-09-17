#include "SigmoidNode.h"

SigmoidNode::SigmoidNode(): ComputationalNode(NT_SIGMOID)
{}

SigmoidNode::~SigmoidNode()
{}

const Eigen::MatrixXd SigmoidNode::Forward(const Eigen::MatrixXd & input)
{
	return input.unaryExpr(&Sigmoid);
}

Eigen::MatrixXd SigmoidNode::Backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd & derivative)
{
	return input.unaryExpr(&Derivative).asDiagonal() * derivative;
}

double SigmoidNode::Sigmoid(double input)
{
	return 1.0 / (1.0 + exp(input));
}

double SigmoidNode::Derivative(double input)
{
	return Sigmoid(input)*(1.0 - Sigmoid(input));
}
