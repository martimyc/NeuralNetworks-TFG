#ifndef FULLY_CONNECTED_LAYER_NODE
#define FULLY_CONNECTED_LAYER_NODE

#include "ComputationNode.h"

class FullyConnectedLayerNode : public virtual ComputationNode
{
public:
	FullyConnectedLayerNode() {}
	virtual ~FullyConnectedLayerNode() {}

	virtual void Forward(Eigen::VectorXd& input) const = 0;
	virtual void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) = 0; // Chain rule

	// UI
	virtual bool UINode() const = 0;
	virtual void UIDescription() const = 0;
};


#endif // !FULLY_CONNECTED_LAYER_NODE

