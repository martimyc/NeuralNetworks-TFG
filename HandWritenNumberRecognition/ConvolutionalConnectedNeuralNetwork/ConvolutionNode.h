#ifndef CONVOLUTION_NODE
#define CONVOLUTION_NODE

#include "ConvolutionLayerNode.h"

class ConvolutionNode : public ConvolutionLayerNode
{
public:
	struct Kernel;

public:
	ConvolutionNode(std::vector<ConvolutionNode::Kernel>& kernels);
	ConvolutionNode(ConvolutionNode::Kernel& kernel);
	~ConvolutionNode();

	void Forward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& output) const override;
	void Backward(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::MatrixXd>& gradients, std::vector<Eigen::MatrixXd>& output) override;

	void UpdateWeightsAndBiases(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float training_rate);
	void UpdateWeightsAndBiasesRegular(const std::vector<Eigen::MatrixXd> & gradients, const std::vector<Eigen::MatrixXd> & inputs, float eta, int mini_batch_size, float lambda);

private:
	double ComputeForwardPixel(int x, int y, const Kernel& kernel, const Eigen::MatrixXd& input) const;
	void ComputeBackwardPixel(int x, int y, const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& input, const Kernel& kernel, Eigen::MatrixXd & output) const;
	void ComputeWeightGradient(int x, int y, const Eigen::MatrixXd & gradient, const Eigen::MatrixXd& input_gradient, const Kernel& kernel, Eigen::MatrixXd& output);

public:
	struct Kernel
	{
		Eigen::MatrixXd weights;
		double bias;

		inline int Rows() const { return weights.rows(); }
		inline int Cols() const { return weights.cols(); }
		inline double Bias() const { return bias; }
		inline const Eigen::MatrixXd& Weights() const { return weights; }
	};

private:
	std::vector<Kernel> kernels;
};

#endif // !CONVOLUTION_NODE

