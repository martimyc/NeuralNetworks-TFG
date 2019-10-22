#ifndef COMPUTATION_NODE
#define COMPUTATION_NODE

#include "Eigen/Dense"
#include <string>

#define BUTTON_SIZE ImVec2(80.0f, 60.0f)

enum NODE_TYPE
{
	NT_SIGMOID,
	NT_FULLY_CONNECTED,
	NT_CONVOLUTION,
	NT_L2_POOLING,
	NT_MAX_POOLING,
	NT_RELU,
	NT_TANH,
	NT_SOFTMAX
};

class ComputationNode
{
public:
	ComputationNode(NODE_TYPE type): type(type) {}
	virtual ~ComputationNode() {}

	virtual bool UINode() const = 0;
	virtual void UIDescription() const = 0;

	inline NODE_TYPE Type() const { return type; }

protected:
	NODE_TYPE type;
};


#endif // !COMPUTATION_NODE

