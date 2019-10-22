#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include "Layer.h"

class FullyConnectedLayerNode;
class FullyConnectedNode;

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int num_neurons, int previous_layer_neurons, ACTIVATION_FUNCTION activation_funct, bool regularization = false);
	~FullyConnectedLayer();

	// Work
	const Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input) override;
	const Eigen::MatrixXd BackPropagate(const Eigen::MatrixXd& gradient, float eta, float mini_batch_size, float lambda = 0.0f) const override; // Need to solve training
	void CleanUp() override;

	// Initialization
	static Eigen::VectorXd RandomInitBias(int num_neurons);
	static Eigen::MatrixXd RandomInitWeight(int num_neurons, int previous_layer_neurons);

	// UI
	void UI() override;

	// Add Nodes
	void AddFullyConnectedNode( int num_neurons, int num_conections);
	void AddReluNode();
	void AddSigmoidNode();
	void AddTanhNode();
	void AddSoftMaxNode();

	// Debug
	void Debug();

	// Getters
	const Eigen::MatrixXd& GetWeights() const;
	const Eigen::MatrixXd& GetBiases() const;

private:
	void MatToVec(const Eigen::MatrixXd& input, Eigen::VectorXd& output) const;
	void VecToMat(const Eigen::VectorXd& input, Eigen::MatrixXd& output) const;

private:
	// Nodes
	std::vector<FullyConnectedLayerNode*> nodes;

	// Inputs for back prop chain rule
	std::vector<Eigen::VectorXd> inputs;

	// Update
	FullyConnectedNode* weights_node;

	// Info
	int num_neurons;
	int num_weights;

	// UI
	FullyConnectedLayerNode* focused;
};


#endif //!FULLY_CONNECTED_LAYER
