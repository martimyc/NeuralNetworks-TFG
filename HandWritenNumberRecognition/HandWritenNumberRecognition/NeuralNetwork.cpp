#include "NeuralNetwork.h"

#include "Neuron.h"
#include "SigmoidNeuron.h"



NeuralNetwork::NeuralNetwork(int input, const std::vector<int>& hidden, int output)
{
	// Input
	input_layer.reserve(input);
	for (int i = 0; i < input; i++)
	{
		input_layer.push_back(Neuron());
	}

	std::vector<Neuron>& connection_layer = input_layer;

	// Hidden Layers
	hidden_layers.reserve(hidden.size());
	for (std::vector<int>::const_iterator it = hidden.begin(); it != hidden.end(); ++it)
	{
		// Add hidden layer
		hidden_layers.push_back(std::vector<Neuron>());
		std::vector<Neuron>& new_layer = hidden_layers.back();
		
		//Fill Hidden layer
		new_layer.reserve(*it);
		for (int i = 0; i < *it; i++)
		{
			new_layer.push_back(Neuron(connection_layer));
		}
		connection_layer = new_layer;
	}

	// Output
	input_layer.reserve(input);
	for (int i = 0; i < input; i++)
	{
		input_layer.push_back(Neuron(connection_layer));
	}
}


NeuralNetwork::~NeuralNetwork()
{
}
