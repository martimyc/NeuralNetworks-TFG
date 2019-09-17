#ifndef FULLY_CONNECTED_NODE
#define FULLY_CONNECTED_NODE

#include "ComputationalNode.h"

class FullyConnectedNode: public ComputationalNode
{
public:
	FullyConnectedNode(int* input, int* output_neurons);
	~FullyConnectedNode();

	const Eigen::MatrixXd Forward(const Eigen::MatrixXd& input) override;
	Eigen::MatrixXd Backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& derivative)  override;

	void UpdateWeightsAndBiases(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float training_rate);
	void UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float eta, int mini_batch_size, float lambda);

private:
	int input_neurons[3];
	int output_neurons[3];
	Eigen::MatrixXd weights;
	Eigen::MatrixXd biases;
};

#endif // !FULLY_CONNECTED_NODE

