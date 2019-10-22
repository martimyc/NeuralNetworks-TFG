#include "TanhNode.h"

#include "imgui.h"

TanhNode::TanhNode(): ComputationNode(NT_TANH)
{}

TanhNode::~TanhNode()
{}

void TanhNode::Forward(Eigen::VectorXd & input) const
{
	input = input.unaryExpr(&Tanh);
}

void TanhNode::Backward(const Eigen::VectorXd & input, Eigen::VectorXd & gradient)
{
	gradient = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void TanhNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	for (std::vector<Eigen::MatrixXd>::iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		*input = input->unaryExpr(&Tanh);
	}
}

void TanhNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
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

double TanhNode::Tanh(double input)
{
	return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
}

double TanhNode::Derivative(double input)
{
	return 1.0 - Tanh(input) * Tanh(input);
}

bool TanhNode::UINode() const
{
	return ImGui::Button("Tahn\nNode", BUTTON_SIZE);
}

void TanhNode::UIDescription() const
{
	ImGui::TextWrapped("Applies the Tanh activation function.");
	// TODO ImGui::Image();
}