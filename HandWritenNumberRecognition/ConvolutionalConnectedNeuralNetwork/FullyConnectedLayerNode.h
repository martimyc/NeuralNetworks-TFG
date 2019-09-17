#ifndef FULLY_CONNECTED_LAYER_NODE
#define FULLY_CONNECTED_LAYER_NODE

#include "ComputationNode.h"

class FullyConnectedLayerNode : public virtual ComputationNode
{
public:
	FullyConnectedLayerNode() {}
	virtual ~FullyConnectedLayerNode() {}

	virtual void Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const = 0;
	virtual void Backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradient, Eigen::VectorXd& output) = 0; // Chain rule
};


#endif // !FULLY_CONNECTED_LAYER_NODE

