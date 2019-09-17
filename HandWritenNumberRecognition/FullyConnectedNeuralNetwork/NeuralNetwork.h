#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>
#include <random>

#include "Eigen/Dense"

class Layer;
class MNIST;

enum STATE {
	S_READY,
	S_TRAINING,
	S_DONE
};

enum COST_FUNCTION {
	CF_QUADRATIC = 0,
	CF_CROSS_ENTHROPY
};

class NeuralNetwork
{
public:
	// Constructor & Destructor
	NeuralNetwork(int input, const std::vector<int>& hidden, int output, COST_FUNCTION cost_function = CF_QUADRATIC, bool regularization = false);
	virtual ~NeuralNetwork();

	// Work
	const Eigen::VectorXd& FeedForward(const Eigen::VectorXd& input);
	void BackPropagation(const Eigen::VectorXd& error, const Eigen::VectorXd & input, float training_rate, int mini_batch_size, float lambda);
	void SGD(const std::vector<MNIST*>& training_data, int epochs, int mini_batch_size, float eta, float lambda = 0.0000f);

	// Getters
	int GetResult(const Eigen::VectorXd& output) const;
	Layer& GetLastLayer();
	inline const STATE& GetState() const { return state; }

	// Cost functions
	Eigen::VectorXd Delta(const Eigen::VectorXd& activation, const Eigen::VectorXd& desired); // Computes overall cost based on the diferent cost functions
	static Eigen::VectorXd Quadratic(const Eigen::VectorXd& activation, const Eigen::VectorXd& desired);
	static Eigen::VectorXd CrossEntropy(const Eigen::VectorXd& activation, const Eigen::VectorXd& desired);

	// UI
	void Info() const;

private:
	void UpdateWithMiniBatch(std::vector<MNIST*>& mini_batch, float eta, float lambda);

	// Test
	void TestOnValidation();
	void TestOnTest();
	void TestOnTraining();

	// Analytics
	float Cost(const Eigen::VectorXd& output, const Eigen::VectorXd& desired);

private:
	std::vector<Layer*> layers;
	int num_inputs;
	STATE state;
	COST_FUNCTION cost_function;
	bool regularization;
};

#endif //!NEURALNETWORK