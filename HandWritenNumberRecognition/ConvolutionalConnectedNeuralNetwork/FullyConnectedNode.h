#ifndef FULLY_CONNECTED_NODE
#define FULLY_CONNECTED_NODE

#include "FullyConnectedLayerNode.h"
#include <random>

class FullyConnectedNode: public FullyConnectedLayerNode
{
public:
	FullyConnectedNode(std::mt19937& engine, int num_neurons, int num_connections);
	FullyConnectedNode(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& biases);
	~FullyConnectedNode();

	void Forward(Eigen::VectorXd& input) const override;
	void Backward(const Eigen::VectorXd& input, Eigen::VectorXd& gradient)  override;

	void UpdateWeightsAndBiases(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float training_rate);
	void UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, float eta, int mini_batch_size, float lambda);

	// UI
	bool UINode() const override;
	void UIDescription() const override;

	// Getters
	inline const Eigen::MatrixXd& GetWeights() const { return weights; }
	inline const Eigen::MatrixXd& GetBiases() const { return biases; }

private:
	Eigen::MatrixXd weights;
	Eigen::MatrixXd biases;
	Eigen::MatrixXd z;
};

#endif // !FULLY_CONNECTED_NODE

