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

private:
	NeuralNetwork* network;

	int output;

	std::thread* training_thread;

	int training_sesion;
};

#endif // !ML_
