#include "L2PoolingNode.h"

#include "imgui.h"

L2PoolingNode::L2PoolingNode() : ComputationNode(NT_L2_POOLING)
{}

L2PoolingNode::~L2PoolingNode()
{}

void L2PoolingNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		Eigen::MatrixXd mat (input->rows() / 2, input->cols() / 2);

		for (int h = 0; h < input->cols() - 1; h += 2)
		{
			for (int w = 0; w < input->rows() - 1; w += 2)
			{
				mat(w/2, h/2) = input->block(w, h, 2, 2).norm();
			}
		}
	}
}

void L2PoolingNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
{
	std::vector<Eigen::MatrixXd>::iterator gradient = gradients.begin();
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

	for (; gradient != gradients.end(); gradient++, input++)
	{
		Eigen::MatrixXd output(Eigen::MatrixXd::Zero(gradient->rows() * 2, gradient->cols() * 2));

		for (int i = 0; i < gradient->rows(); i++)
		{
			for (int j = 0; j < gradient->cols(); j++)
			{
				Eigen::MatrixXd mat(2, 2);
				mat(0, 0) = (*input)(i * 2, j * 2);
				mat(0, 1) = (*input)(i * 2, j * 2 + 1);
				mat(1, 0) = (*input)(i * 2 + 1, j * 2);
				mat(1, 1) = (*input)(i * 2 + 1, j * 2 + 1);

				if (mat.maxCoeff() == mat(0, 0))
				{
					output(i * 2, j * 2) = (*gradient)(i, j);
				}
				if (mat.maxCoeff() == mat(0, 1))
				{
					output(i * 2, j * 2 + 1) = (*gradient)(i, j);
				}
				if (mat.maxCoeff() == mat(1, 0))
				{
					output(i * 2 + 1, j * 2) = (*gradient)(i, j);
				}
				if (mat.maxCoeff() == mat(1, 1))
				{
					output(i * 2 + 1, j * 2 + 1) = (*gradient)(i, j);
				}
			}
		}
		*gradient = output;
	}
}

bool L2PoolingNode::UINode() const
{
	return ImGui::Button("L2\nPooling\nNode", BUTTON_SIZE);
}

void L2PoolingNode::UIDescription() const
{
	ImGui::TextWrapped("Pooling nodes simplify the input.\nin the L2 case the output is the square root of a nuber of inputs squared.");
}
