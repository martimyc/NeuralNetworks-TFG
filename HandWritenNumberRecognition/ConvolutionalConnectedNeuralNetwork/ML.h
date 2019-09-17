#ifndef ML_
#define ML_

#include <vector>
#include <thread>
#include <random>

#include "Module.h"

class NeuralNetwork;

class ML: public Module
{
public:
	ML();
	~ML();

	// Functionality
	bool Init() override;
	bool Start() override;
	bool PreUpdate() override;
	bool Update() override;
	bool PostUpdate() override;
	bool CleanUp() override;

	// Layers
	void AddFullyConnectedLayer(int num_neurons, int num_previous_layer_neurons);

private:
	NeuralNetwork* network;

	int output;

	std::thread* training_thread;

	bool training = false;
	int training_sesion;

	// Random
	std::random_device rand_device;
	std::mt19937 engine;
};

#endif // !ML_
