#ifndef SIGMOID
#define SIGMOID

#include "ComputationalNode.h"

class SigmoidNode : ComputationalNode
{
public:
	SigmoidNode();
	~SigmoidNode();

	const Eigen::MatrixXd Forward(const Eigen::MatrixXd& input) override;
	Eigen::MatrixXd Backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& derivative) override;

	static double Sigmoid(double input);
	static double Derivative(double input);
};


#endif // !SIGMOID

