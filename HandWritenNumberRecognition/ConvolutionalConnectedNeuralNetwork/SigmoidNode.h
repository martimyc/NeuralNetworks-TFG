#ifndef SIGMOID
#define SIGMOID

#include "ConvolutionLayerNode.h"
#include "FullyConnectedLayerNode.h"

class SigmoidNode : public ConvolutionLayerNode, public FullyConnectedLayerNode
{
public:
	SigmoidNode();
	~SigmoidNode();

	// Fully Connected
	void Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
	void Backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradient, Eigen::VectorXd& output) override;

	// Convolution
	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;

	static double Sigmoid(double input);
	static double Derivative(double input);
};


#endif // !SIGMOID

