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
	void Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
	void Backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradient, Eigen::VectorXd& output) override;

	// Convolution
	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;

	static double Relu(double input);
	static double Derivative(double input);
};

#endif //!RELU_NODE
