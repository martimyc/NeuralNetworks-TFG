#ifndef SOFTMAX_NODE
#define SOFTMAX_NODE

#include "FullyConnectedLayerNode.h"
#include <random>

class SoftMaxNode : public FullyConnectedLayerNode
{
public:
	SoftMaxNode();
	~SoftMaxNode();

	// Fully Connected
	void Forward(Eigen::VectorXd& input) const override;
	void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) override;

	static double SoftMax(double input, const Eigen::VectorXd& z);
	static double Derivative(double input, const Eigen::VectorXd& z);

	// UI
	bool UINode() const override;
	void UIDescription() const override;
};

#endif // !SOFTMAX_NODE