#ifndef RELU_NODE
#define RELU_NODE

#include "ConvolutionLayerNode.h"
#include "FullyConnectedLayerNode.h"

class ReluNode : public ConvolutionLayerNode, public FullyConnectedLayerNode
{
public:
	ReluNode();
	~ReluNode();

	// Fully Connected
	void Forward(Eigen::VectorXd& input) const override;
	void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) override;

	// Convolution
	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	static double Relu(double input);
	static double Derivative(double input);

	// UI
	bool UINode() const override;
	void UIDescription() const override;
};

#endif //!RELU_NODE
