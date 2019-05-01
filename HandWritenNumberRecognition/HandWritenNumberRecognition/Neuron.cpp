#include "Neuron.h"

Neuron::Neuron()
{}

Neuron::Neuron(std::vector<Neuron>& connected)
{
	// Reserve space
	connections.reserve(connected.size());

	// Connect to previous layer
	for (std::vector<Neuron>::iterator it = connected.begin(); it != connected.end(); ++it)
	{
		connections.push_back(Connection(*it, 0.0f));
	}
}

Neuron::~Neuron()
{
}