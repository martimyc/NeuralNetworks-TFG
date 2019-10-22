#ifndef MAX_POOLING_NODE
#define MAX_POOLING_NODE

#include "ConvolutionLayerNode.h"

class MaxPoolingNode : public ConvolutionLayerNode
{
public:
	MaxPoolingNode();
	~MaxPoolingNode();

	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	// UI
	bool UINode() const override;
	void UIDescription() const override;

private:
	int window_h;
	int window_w;
};

#endif // !MAX_POOLING_NODE

