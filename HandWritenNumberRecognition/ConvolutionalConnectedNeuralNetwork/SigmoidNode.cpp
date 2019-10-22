#include "SigmoidNode.h"

#include "imgui.h"

SigmoidNode::SigmoidNode(): ComputationNode(NT_SIGMOID)
{}

SigmoidNode::~SigmoidNode()
{}

void SigmoidNode::Forward(Eigen::VectorXd & input) const
{
	input = input.unaryExpr(&Sigmoid);
}

void SigmoidNode::Backward(const Eigen::VectorXd & input, Eigen::VectorXd & gradient)
{
	gradient = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void SigmoidNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	for (std::vector<Eigen::MatrixXd>::iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		*input = input->unaryExpr(&Sigmoid);
	}
}

void SigmoidNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
{
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();
	std::vector<Eigen::MatrixXd>::iterator gradient = gradients.begin();
	for (; input != inputs.end(); input++, gradient++)
	{
		for (int i = 0; i < input->rows(); i++)
		{
			for (int j = 0; j < input->cols(); j++)
			{
				(*gradient)(i, j) = Derivative((*input)(i, j)) * (*gradient)(i, j);
			}
		}
	}
}

double SigmoidNode::Sigmoid(double input)
{
	return 1.0 / (1.0 + exp(-input));
}

double SigmoidNode::Derivative(double input)
{
	return Sigmoid(input)*(1.0 - Sigmoid(input));
}

bool SigmoidNode::UINode() const
{
	return ImGui::Button("Sigmoid\nNode", BUTTON_SIZE);
}

void SigmoidNode::UIDescription() const
{
	ImGui::TextWrapped("Applies the Sigmoid activation function.");
	// TODO ImGui::Image();
}