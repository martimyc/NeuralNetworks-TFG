#include "L2PoolingNode.h"

L2PoolingNode::L2PoolingNode() : ComputationNode(NT_L2_POOLING)
{}

L2PoolingNode::~L2PoolingNode()
{}

void L2PoolingNode::Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const
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
				mat(w, h) = input->block(w * 2, h * 2, 2, 2).norm();
			}
		}
	}
}

void L2PoolingNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output)
{
	// TODO
}