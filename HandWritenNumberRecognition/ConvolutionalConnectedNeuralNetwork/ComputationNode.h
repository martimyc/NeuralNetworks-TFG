#ifndef COMPUTATION_NODE
#define COMPUTATION_NODE

#include "Eigen/Dense"

enum NODE_TYPE
{
	NT_SIGMOID,
	NT_FULLY_CONNECTED,
	NT_CONVOLUTION,
	NT_L2_POOLING,
	NT_MAX_POOLING,
	NT_RELU,
	NT_TANH
};

class ComputationNode
{
public:
	ComputationNode(NODE_TYPE type): type(type) {}
	virtual ~ComputationNode() {}

	inline NODE_TYPE Type() const { return type; }

protected:
	NODE_TYPE type;
};


#endif // !COMPUTATION_NODE

