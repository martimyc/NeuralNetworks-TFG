#ifndef COMPUTATIONAL_NODE
#define COMPUTATIONAL_NODE

#include "Eigen/Dense"

enum NODE_TYPE
{
	NT_SIGMOID,
	NT_FULLY_CONNECTED
};

class ComputationalNode
{
public:
	ComputationalNode(NODE_TYPE type): type(type) {}
	~ComputationalNode() {}

	virtual const Eigen::MatrixXd Forward(const Eigen::MatrixXd& input) = 0;
	virtual Eigen::MatrixXd Backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& derivative) = 0; // Will go back on last computed forward, muct be called after

protected:
	NODE_TYPE type;
};


#endif // !COMPUTATIONAL_NODE

