#ifndef CONVOLUTION_LAYER
#define CONVOLUTION_LAYER

#include "Layer.h"
#include <vector>

class ConvolutionLayerNode;
class ConvolutionNode;

class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(int k_size, POOLING pooling, ACTIVATION_FUNCTION activation_function, int num_filters, int input_image_size, bool regularization = false);
	~ConvolutionLayer();

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
	void AddConvolutionNode(int k_size, int num_filters);
	void AddL2PoolingNode();
	void AddMaxPoolingNode();
	void AddReluNode();
	void AddSigmoidNode();
	void AddTanhNode();

	// Debug
	void Debug();

	// Getters
	inline int GetOutputImageSize() const { return output_image_h * output_image_w; }

private:
	void MatToVecFw(const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& output) const;
	void MatToVecBack(const Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& output) const;
	void VecToMat(const std::vector<Eigen::MatrixXd>& input, Eigen::MatrixXd& output) const;

private:
	// Info
	int k_size;
	POOLING pooling;
	int num_filters;
	int input_image_h;
	int input_image_w;
	int output_image_h;
	int output_image_w;
	
	// Nodes
	std::vector<ConvolutionLayerNode*> nodes;

	// Inputs for back prop chain rule
	std::vector<std::vector<Eigen::MatrixXd>> inputs;

	// Update
	ConvolutionNode* convolution_node;

	// UI
	ConvolutionLayerNode* focused;
};

#endif // !CONVOLUTION_LAYER

CONVOLUTION_LAYER

