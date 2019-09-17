#include "ML.h"

#include <string>
#include <iostream>

#include "imgui.h"

#include "Globals.h"
#include "NeuralNetwork.h"
#include "Application.h"
#include "Datasets.h"
#include "MNIST.h"


ML::ML() : Module("ML"), network(nullptr), training_thread(nullptr), training_sesion(0)
{}

ML::~ML()
{}

bool ML::Init()
{
	// Initialize NN
	std::vector<int> hidden_layers;
	hidden_layers.reserve(NUM_HIDDEN_LAYERS);

	hidden_layers.push_back(HIDDEN_LAYER_1);

	network = new NeuralNetwork(INPUT_LAYER, hidden_layers, OUTPUT_LAYER, CF_CROSS_ENTHROPY, true);

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