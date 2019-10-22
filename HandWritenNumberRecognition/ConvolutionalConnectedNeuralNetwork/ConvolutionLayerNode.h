#ifndef CONVOLUTION_LAYER_NODE
#define CONVOLUTION_LAYER_NODE

#include <vector>
#include "ComputationNode.h"

class ConvolutionLayerNode : public virtual ComputationNode
{
public:
	ConvolutionLayerNode() {}
	virtual ~ConvolutionLayerNode() {}

	virtual void Forward(std::vector<Eigen::MatrixXd>& inputs) const = 0;
	virtual void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradient) = 0; // Chain rule

	// UI
	virtual bool UINode() const = 0;
	virtual void UIDescription() const = 0;
};


#endif // !CONVOLUTION_LAYER_NODE