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
	void Forward(Eigen::VectorXd& input) const override;
	void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) override;

	// Convolution
	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	static double Sigmoid(double input);
	static void MatrixSigmoid(Eigen::MatrixXd& input);
	static double Derivative(double input);
	static void MatrixDerivative(Eigen::MatrixXd& input);

	// UI
	bool UINode() const override;
	void UIDescription() const override;
};


#endif // !SIGMOID

