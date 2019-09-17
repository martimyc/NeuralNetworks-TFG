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
	void Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
	void Backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradient, Eigen::VectorXd& output) override;

	// Convolution
	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;

	static double Tanh(double input);
	static double Derivative(double input);
};

#endif // !TANH_NODE
