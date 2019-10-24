#include "ML.h"

#include <string>
#include <iostream>

#include "imgui.h"

#include "Globals.h"
#include "NeuralNetwork.h"
#include "Application.h"
#include "Datasets.h"
#include "Analytics.h"
#include "MNIST.h"
#include "Layer.h"
#include "ConvolutionLayer.h"

ML::ML() : Module("ML"), network(nullptr), training_thread(nullptr)
{}

ML::~ML()
{}

bool ML::Init()
{
	// Initialize NN
	network = new NeuralNetwork(CF_CROSS_ENTHROPY, true);
	
	//network->AddFullyConnectedLayer(30, INPUT_LAYER, AF_SIGMOID, false);
	//network->AddFullyConnectedLayer(10, 30, AF_SOFTMAX, false);

	// Convolutional
	ConvolutionLayer* conv_layer = network->AddConvolutionLayer(4, P_MAX, AF_RELU, 4, IMAGE_SIZE, true);
	conv_layer = network->AddConvolutionLayer(4, P_MAX, AF_RELU, 4, conv_layer->GetOutputImageSize(), true);

	// Fully Connected
	// neurons to fully connect
	// Image size of last layer * number of filters of layers before (4 in the first 4 in the second)
	int num_neurons = conv_layer->GetOutputImageSize() * 4 * 4; 

	network->AddFullyConnectedLayer(30, num_neurons, AF_RELU);
	network->AddFullyConnectedLayer(16, 30, AF_RELU, false, true);
	network->AddFullyConnectedLayer(10, 16, AF_SOFTMAX, true, true);

	// Debug
	//network->AddFullyConnectedLayer(30, 4*4, AF_SIGMOID, false);
	//network->AddFullyConnectedLayer(10, 30, AF_SIGMOID, false);

	/*network->AddFullyConnectedLayer(30, IMAGE_SIZE, AF_SIGMOID, false, true);
	network->AddFullyConnectedLayer(16, 30, AF_SIGMOID, false, true);
	network->AddFullyConnectedLayer(10, 16, AF_SOFTMAX, true, true);*/

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

	if (ImGui::TreeNodeEx("Training", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
	{
		ImGui::Text("Training Sesion %i", network->GetTrainingSesion());

		ImGui::Text("Training rate: %f", TRAINING_RATE);

		ImGui::Text("Epochs: %i", EPOCHS);

		ImGui::Text("Mini Batch Size: %i", MINI_BATCH_SIZE);

		ImGui::Text("Regularization parameter: %f", REGULARIZATION_PARAMETER);

		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("Network", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
	{
		network->Info();

		ImGui::TreePop();
	}

	ImGui::End();

	if (training_thread == nullptr && App->dataset->LoadingDone())
	{
		InitNetworkTraining();
	}

	/*network->DebugConvolution();
	network->DebugLayer();
	network->Debug();*/

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
		network->Stop();
		training_thread->join();
		delete training_thread;
		training_thread = nullptr;
	}

	delete network;

	return true;
}

void ML::InitNetworkTraining()
{
	training_thread = new std::thread(&NeuralNetwork::SGD, std::ref(*network), App->dataset->GetTrainingSet(), EPOCHS, MINI_BATCH_SIZE, TRAINING_RATE, REGULARIZATION_PARAMETER);
	App->analitycs->StartTimer();
}
