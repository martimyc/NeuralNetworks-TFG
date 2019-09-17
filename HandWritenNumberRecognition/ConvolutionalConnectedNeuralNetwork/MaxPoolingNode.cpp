#include "MaxPoolingNode.h"

MaxPoolingNode::MaxPoolingNode() : ComputationNode(NT_MAX_POOLING)
{}

MaxPoolingNode::~MaxPoolingNode()
{}

void MaxPoolingNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
{
	output.reserve(inputs.size());
	for (std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin(); input != inputs.end(); input++)
	{
		output.push_back(Eigen::MatrixXd(input->rows() / 2, input->cols() / 2));
		Eigen::MatrixXd& mat = output.back();

		for (int h = 0; h < input->cols(); h++)
		{
			for (int w = 0; w < input->rows(); w++)
			{
				mat(w, h) = input->block(w * 2, h * 2, 2, 2).maxCoeff();
			}
		}
	}
}

void MaxPoolingNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
{
	output.reserve(gradients.size());
	{
		std::vector<Eigen::MatrixXd>::const_iterator gradient = gradients.begin();
		std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

		for (; gradient != gradients.end(); gradient++, input++)
		{
			output.push_back(Eigen::MatrixXd::Zero(gradient->rows() * 2, gradient->cols() * 2));
			Eigen::MatrixXd& mat = output.back();

			for (int i = 0; i < gradient->rows(); i++)
			{
				for (int j = 0; j < gradient->cols(); j++)
				{
					double input00 = (*input)(i * 2, j * 2);
					double input01 = (*input)(i * 2, j * 2 + 1);
					double input10 = (*input)(i * 2 + 1, j * 2);
					double input11 = (*input)(i * 2 + 1, j * 2 + 1);

					if (input00 > input01 && input00 > input10 && input00 > input11)
					{
						mat(i * 2, j * 2) = (*gradient)(i, j);
					}
					else if (input01 > input00 && input01 > input10 && input01 > input11)
					{
						mat(i * 2, j * 2 + 1) = (*gradient)(i, j);
					}
					else if (input10 > input01 && input10 > input00 && input10 > input11)
					{
						mat(i * 2 + 1, j * 2) = (*gradient)(i, j);
					}
					else if (input11 > input01 && input11 > input10 && input11 > input00)
					{
						mat(i * 2 + 1, j * 2 + 1) = (*gradient)(i, j);
					}
						
				}
			}
		}
	}
}