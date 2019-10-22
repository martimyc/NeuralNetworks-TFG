#ifndef TANH_NODE
#define TANH_NODE

#include "ConvolutionLayerNode.h"
#include "FullyConnectedLayerNode.h"

class TanhNode : public ConvolutionLayerNode, public FullyConnectedLayerNode
{
public:
	TanhNode();
	~TanhNode();

	// Fully Connected
	void Forward(Eigen::VectorXd& input) const override;
	void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) override;

	// Convolution
	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	static double Tanh(double input);
	static double Derivative(double input);

	// UI
	bool UINode() const override;
	void UIDescription() const override;
};

#endif // !TANH_NODE
