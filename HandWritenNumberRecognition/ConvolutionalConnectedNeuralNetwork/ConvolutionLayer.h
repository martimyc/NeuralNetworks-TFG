#ifndef CONVOLUTION_LAYER
#define CONVOLUTION_LAYER

#include "Layer.h"
#include <vector>

class ConvolutionLayerNode;
class ConvolutionNode;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(bool regularization = false);
	~ConvolutionLayer();

	// Work
	const Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input) override;
	const Eigen::MatrixXd BackPropagate(const Eigen::MatrixXd& gradient, float eta, float mini_batch_size, float lambda = 0.0f) const override; // Need to solve training

	// Initialization
	static Eigen::VectorXd RandomInitBias(int num_neurons);
	static Eigen::MatrixXd RandomInitWeight(int num_neurons, int previous_layer_neurons);

	// TODO Add Nodes

private:
	// TODO
	void MatToVec(const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& output) const;
	void VecToMat(const std::vector<Eigen::MatrixXd>& input, Eigen::MatrixXd& output) const;

private:
	// Nodes
	std::vector<ConvolutionLayerNode*> nodes;

	// Inputs for back prop chain rule
	std::vector<std::vector<Eigen::MatrixXd>> inputs;

	// Update
	ConvolutionNode* weights_node;
};

#endif // !CONVOLUTION_LAYER

CONVOLUTION_LAYER

