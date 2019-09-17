#ifndef CONVOLUTION_LAYER_NODE
#define CONVOLUTION_LAYER_NODE

#include <vector>
#include "ComputationNode.h"

class ConvolutionLayerNode : public virtual ComputationNode
{
public:
	ConvolutionLayerNode() {}
	virtual ~ConvolutionLayerNode() {}

	virtual void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const = 0;
	virtual void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradient, std::vector<Eigen::MatrixXd>& output) = 0; // Chain rule
};


#endif // !CONVOLUTION_LAYER_NODE