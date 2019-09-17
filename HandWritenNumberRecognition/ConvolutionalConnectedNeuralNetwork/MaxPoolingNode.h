#ifndef MAX_POOLING_NODE
#define MAX_POOLING_NODE

#include "ConvolutionLayerNode.h"

class MaxPoolingNode : public ConvolutionLayerNode
{
public:
	MaxPoolingNode();
	~MaxPoolingNode();

	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;
};

#endif // !MAX_POOLING_NODE

