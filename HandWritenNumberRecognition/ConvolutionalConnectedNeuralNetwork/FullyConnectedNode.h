#ifndef FULLY_CONNECTED_NODE
#define FULLY_CONNECTED_NODE

#include "FullyConnectedLayerNode.h"

class FullyConnectedNode: public FullyConnectedLayerNode
{
public:
	FullyConnectedNode(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& biases);
	~FullyConnectedNode();

	void Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const override;
	void Backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradient, Eigen::VectorXd& output)  override;

	void UpdateWeightsAndBiases(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float training_rate);
	void UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float eta, int mini_batch_size, float lambda);

private:
	Eigen::MatrixXd weights;
	Eigen::MatrixXd biases;
};

#endif // !FULLY_CONNECTED_NODE

