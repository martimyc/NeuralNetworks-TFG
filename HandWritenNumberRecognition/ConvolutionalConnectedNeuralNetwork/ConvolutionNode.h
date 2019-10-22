#ifndef CONVOLUTION_NODE
#define CONVOLUTION_NODE

#include "ConvolutionLayerNode.h"
#include <random>

class ConvolutionNode : public ConvolutionLayerNode
{
public:
	class Kernel;

public:
	ConvolutionNode(std::mt19937 & engine, int k_size, int num_filters);
	~ConvolutionNode();

	void Forward(std::vector<Eigen::MatrixXd>& inputs) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients) override;

	void UpdateWeightsAndBiases(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float training_rate);
	void UpdateWeightsAndBiasesRegular(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float eta, int mini_batch_size, float lambda);

	// UI
	bool UINode() const override;
	void UIDescription() const override;

private:
	double ComputeForwardPixel(int x, int y, const Kernel& kernel, const Eigen::MatrixXd& input) const;
	void ComputeBackwardPixel(int x, int y, const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, const Kernel& kernel, Eigen::MatrixXd & output) const;
	void ComputeWeightGradient(int x, int y, const Eigen::MatrixXd & gradient, const Eigen::MatrixXd& input_gradient, const Kernel& kernel, Eigen::MatrixXd& output);

public:
	class Kernel
	{
	public:
		Eigen::MatrixXd weights;
		double bias;

	public:
		Kernel(int k_size): weights(k_size, k_size)
		{}

		inline int Rows() const { return weights.rows(); }
		inline int Cols() const { return weights.cols(); }
		inline double Bias() const { return bias; }
		inline const Eigen::MatrixXd& Weights() const { return weights; }
	};

private:
	std::vector<Kernel> kernels;
};

#endif // !CONVOLUTION_NODE

