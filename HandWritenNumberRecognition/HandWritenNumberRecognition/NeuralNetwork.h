#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <vector>

class Neuron;

class NeuralNetwork
{
public:
	NeuralNetwork(int input, const std::vector<int>& hidden, int output);
	virtual ~NeuralNetwork();

protected:
	std::vector<Neuron> input_layer;
	std::vector<std::vector<Neuron>> hidden_layers;
	std::vector<Neuron> output_layer;
};

#endif //!NEURALNETWORK