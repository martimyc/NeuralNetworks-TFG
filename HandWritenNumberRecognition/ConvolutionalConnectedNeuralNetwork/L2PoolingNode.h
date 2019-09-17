#ifndef L2_POOLING_NODE
#define L2_POOLING_NODE

#include "ConvolutionLayerNode.h"

class L2PoolingNode : public ConvolutionLayerNode
{
public:
	L2PoolingNode();
	~L2PoolingNode();

	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;

};

#endif // !L2_POOLING_NODE

