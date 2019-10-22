#include "SoftMaxNode.h"

#include "imgui.h"

SoftMaxNode::SoftMaxNode() : ComputationNode(NT_SOFTMAX)
{}

SoftMaxNode::~SoftMaxNode()
{}

void SoftMaxNode::Forward(Eigen::VectorXd & input) const
{
	Eigen::VectorXd output(Eigen::VectorXd::Zero(input.size()));
	for (int i = 0; i < input.size(); i++)
	{
		output(i) = SoftMax(input(i), input);
	}
	input = output;
}

void SoftMaxNode::Backward(const Eigen::VectorXd & input, Eigen::VectorXd & gradient)
{	
	/*Eigen::VectorXd prime(Eigen::VectorXd::Zero(input.size()));
	for (int i = 0; i < input.size(); i++)
	{
		prime(i) = Derivative(input(i), input);
	}
	gradient = prime.asDiagonal() * gradient;*/
}

double SoftMaxNode::SoftMax(double input, const Eigen::VectorXd& z)
{
	Eigen::VectorXd exponentials(z.size());
	for (int i = 0; i < z.size(); i++)
	{
		exponentials(i) = exp(z(i));
	}
	
	return exp(input) / exponentials.sum();
}

double SoftMaxNode::Derivative(double input, const Eigen::VectorXd& z)
{
	Eigen::VectorXd exponentials(z.size());
	for (int i = 0; i < z.size(); i++)
	{
		exponentials(i) = exp(z(i));
	}

	double c = exponentials.sum() - exp(input);

	return (exp(input) * c) / ((exp(input) + c) * (exp(input) + c));
}

bool SoftMaxNode::UINode() const
{
	return ImGui::Button("SoftMax\nNode", BUTTON_SIZE);
}

void SoftMaxNode::UIDescription() const
{
	ImGui::TextWrapped("Applies the SoftMax activation function.");
	// TODO ImGui::Image();
}