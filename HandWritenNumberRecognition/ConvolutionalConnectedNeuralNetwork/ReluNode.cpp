#include "ReluNode.h"

#include "imgui.h"

ReluNode::ReluNode(): ComputationNode(NT_RELU)
{}

ReluNode::~ReluNode()
{}

void ReluNode::Forward(Eigen::VectorXd & input) const
{
	input = input.unaryExpr(&Relu);
}

void ReluNode::Backward(const Eigen::VectorXd & input, Eigen::VectorXd & gradient)
{
	gradient = input.unaryExpr(&Derivative).asDiagonal() * gradient;
}

void ReluNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	for (std::vector<Eigen::MatrixXd>::iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		*input = input->unaryExpr(&Relu);
	}
}

void ReluNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
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
		return 1;
	}
}

bool ReluNode::UINode() const
{
	return ImGui::Button("RELU\nNode", BUTTON_SIZE);
}

void ReluNode::UIDescription() const
{
	ImGui::TextWrapped("Applies the RELU activation function.");
	// TODO ImGui::Image();
}
