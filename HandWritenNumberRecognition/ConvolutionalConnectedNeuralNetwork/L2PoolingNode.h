#ifndef L2_POOLING_NODE
#define L2_POOLING_NODE

#include "ConvolutionLayerNode.h"

class L2PoolingNode : public ConvolutionLayerNode
{
public:
	L2PoolingNode();
	~L2PoolingNode();

	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	// UI
	bool UINode() const override;
	void UIDescription() const override;
};

#endif // !L2_POOLING_NODE

