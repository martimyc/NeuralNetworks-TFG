#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include "Layer.h"

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int num_neurons, int previous_layer_neurons);
	~FullyConnectedLayer();

	// Work
	const Eigen::VectorXd& FeedForward(const Eigen::MatrixXd& input) override;
	const Eigen::VectorXd BackPropagate(const Eigen::MatrixXd& next_error, const Eigen::MatrixXd& next_weights) const override;
	void UpdateWeightsAndBiases(const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous_activation, float training_rate) override;
	void UpdateWeightsAndBiasesRegular(const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous_activation, float eta, int mini_batch_size, float lambda) override;

	// Initialization
	static Eigen::VectorXd RandomInitBias(int num_neurons);
	static Eigen::MatrixXd RandomInitWeight(int num_neurons, int previous_layer_neurons);

	// Getters
	inline const Eigen::VectorXd& GetZ() override { return z; }
	inline const Eigen::VectorXd& GetActivation() override { return activation_values; }
	inline const Eigen::MatrixXd& GetWeights() override { return weights; }
	inline const int GetNumNeurons() const override { return num_neurons; }

private:
	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;
	Eigen::VectorXd z;
	Eigen::VectorXd activation_values;
	int num_neurons;
	int num_weights;
};


#endif //!FULLY_CONNECTED_LAYER
