#include "ML.h"

#include <string>
#include <iostream>

#include "imgui.h"

#include "Globals.h"
#include "NeuralNetwork.h"
#include "Application.h"
#include "Datasets.h"
#include "MNIST.h"

ML::ML() : Module("ML"), network(nullptr), training_thread(nullptr), training_sesion(0), rand_device(), engine(rand_device())
{}

ML::~ML()
{}

bool ML::Init()
{
	// Initialize NN
	network = new NeuralNetwork(CF_QUADRATIC, false);
	
	AddFullyConnectedLayer(30, INPUT_LAYER);
	AddFullyConnectedLayer(10, 30);

	output = -1;

	return true;
}

bool ML::Start()
{
	return true;
}

bool ML::PreUpdate()
{
	return true;
}

bool ML::Update()
{
	ImGui::Begin("Neural Network Info");

	if (ImGui::TreeNodeEx("Training", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Text("Training Sesion %i", training_sesion);

		ImGui::Text("Training rate: %f", TRAINING_RATE);

		ImGui::Text("Epochs: %i", EPOCHS);

		ImGui::Text("Mini Batch Size: %i", MINI_BATCH_SIZE);

		ImGui::Text("Regularization parameter: %f", REGULARIZATION_PARAMETER);

		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("Network", ImGuiTreeNodeFlags_DefaultOpen))
	{
		network->Info();

		ImGui::TreePop();
	}

	if (!training && App->dataset->LoadingDone())
	{
		training = true;
		training_sesion++;
		training_thread = new std::thread(&NeuralNetwork::SGD, std::ref(*network), App->dataset->GetTrainingSet(), EPOCHS, MINI_BATCH_SIZE, TRAINING_RATE, REGULARIZATION_PARAMETER);
	}

	if (network->GetState() == S_DONE)
	{
		training_thread->join();
		delete training_thread;
		training_thread = nullptr;
		training = false;
	}

	ImGui::End();

	return true;
}

bool ML::PostUpdate()
{
	return true;
}

bool ML::CleanUp()
{
	if (training_thread != nullptr)
	{
		training_thread->join();
		delete training_thread;
		training_thread = nullptr;
		training = false;
	}

	delete network;

	return true;
}

void ML::AddFullyConnectedLayer(int num_neurons, int num_previous_layer_neurons)
{
	// Weigths
	std::normal_distribution<double> distribution_weights(0.0, 1.0 / sqrt(num_previous_layer_neurons));
	Eigen::MatrixXd weights(num_neurons, num_previous_layer_neurons);

	for (int i = 0; i < weights.rows(); i++)
	{
		for (int j = 0; j < weights.cols(); j++)
		{
			weights(i, j) = distribution_weights(engine);
		}
	}

	// Biases
	std::normal_distribution<double> distribution_biases(0.0, 1.0);
	Eigen::MatrixXd biases(num_neurons, 1);

	for (int i = 0; i < biases.rows(); i++)
	{
		weights(i, 1) = distribution_biases(engine);
	}

	//TODO
}
