#ifndef NEURON
#define NEURON

#include <vector>

struct Connection
{
	Connection(Neuron& neuron, float weight) : neuron(neuron), weight(weight)
	{}

	Neuron& neuron;
	float weight;
};

class Neuron
{
public:
	Neuron(); // For non connected input neurons
	Neuron(std::vector<Neuron>& connected);
	virtual ~Neuron();

private:
	float value;
	float bias;
	std::vector<Connection> connections;
};

#endif //!NEURON

