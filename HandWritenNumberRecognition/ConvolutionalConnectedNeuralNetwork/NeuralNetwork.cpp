#include "NeuralNetwork.h"

#include <iostream>
#include <random>
#include <time.h>

#include "imgui.h"

// Modules
#include "Analytics.h"
#include "Application.h"
#include "Datasets.h"

// MNIST
#include "MNIST.h"

// Layers
#include "Layer.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionLayer.h"

NeuralNetwork::NeuralNetwork( COST_FUNCTION cost_function, bool regularization): state(S_READY), cost_function(cost_function), regularization(regularization)
{
	// Seed rand for later
	std::srand(time(NULL));
}

NeuralNetwork::~NeuralNetwork()
{
	for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++)
	{
		delete *layer;
	}
}

const Eigen::MatrixXd& NeuralNetwork::FeedForward(const Eigen::MatrixXd& input) const
{
	Eigen::MatrixXd input_mat(input);

	for (std::vector<Layer*>::const_iterator layer = layers.begin(); layer != layers.end(); layer++)
	{
		input_mat = (*layer)->FeedForward(input_mat);
	}

	return input_mat;
}

void NeuralNetwork::BackPropagation(const Eigen::MatrixXd& cost, const std::vector<Eigen::MatrixXd> & inputs_vec, float eta, int mini_batch_size, float lambda)
{
	Eigen::MatrixXd gradient(cost);

	for (std::vector<Layer*>::const_iterator layer = layers.begin(); layer != layers.end(); layer++)
	{
		gradient = (*layer)->FeedForward(gradient);
	}
}

void NeuralNetwork::SGD(const std::vector<MNIST*>& training_data, int epochs, int mini_batch_size, float eta, float lambda)
{
	state = S_TRAINING;

	// todo paralel

	for (int i = 0; i < epochs; i++)
	{
		// Create random mini batch
		std::vector<MNIST*> mini_batch;
		mini_batch.reserve(mini_batch_size);

		for (int k = 0; k < mini_batch_size; k++)
		{
			int random = std::rand() % training_data.size();
			mini_batch.push_back(training_data[random]);
		}

		// Train with minibatch
		UpdateWithMiniBatch(mini_batch, eta, lambda);

		// Test
		TestOnValidation();
		//TestOnTest();
		//TestOnTraining();
	}

	state = S_DONE;
}

int NeuralNetwork::ComputeResult(Eigen::MatrixXd & input) const
{
	input = FeedForward(input);
	return GetResult(input);
}

// Layers
FullyConnectedLayer * NeuralNetwork::AddFullyConnectedLayer(int layer_neurons, int previous_layer_neurons, bool regularization)
{
	layers.push_back(new FullyConnectedLayer(layer_neurons, previous_layer_neurons, regularization));
	return (FullyConnectedLayer *)layers.back();
}

ConvolutionLayer * NeuralNetwork::AddConvolutionLayer(bool regularization)
{
	layers.push_back(new ConvolutionLayer(regularization));
	return (ConvolutionLayer *)layers.back();
}

// UI
void NeuralNetwork::Info() const
{
	ImGui::Text("Layers: %i", layers.size());

	// testing to show layers
	ImGui::ShowTestWindow();

	/*for (int i = 0; i < layers.size() - 1; i++)
	{
		ImGui::Text("Hidden layer %i neurons: %i", i, layers[i]->GetNumNeurons());
	}*/
}

// Private
// Work
void NeuralNetwork::UpdateWithMiniBatch(std::vector<MNIST*>& mini_batch, float eta, float lambda)
{
	std::vector<Eigen::MatrixXd> inputs;

	for (std::vector<MNIST*>::iterator it = mini_batch.begin(); it != mini_batch.end(); it++)
	{
		// Feed forward
		const Eigen::MatrixXd& image((*it)->GetImage());
		const Eigen::MatrixXd& activation = FeedForward(image);

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(activation.rows(), 1));
		desired((*it)->GetLabel(), 1) = 1.0;

		// With the first layers error we can back propagate and get the next layer's error as well as change its weights and biases
		BackPropagation(Delta(activation, desired), inputs, eta, mini_batch.size(), lambda);
	}
}

// Cost Functions
Eigen::MatrixXd NeuralNetwork::Delta(const Eigen::MatrixXd & activation, const Eigen::MatrixXd & desired)
{
	// Check
	switch (cost_function)
	{
	case CF_QUADRATIC: return desired - activation;
	case CF_CROSS_ENTHROPY: return desired - activation;

	default:
		std::cerr << "NeuralNetwork - Delta - Unknown cost function" << std::endl;
		return Eigen::MatrixXd();
	}
}

// Test
void NeuralNetwork::TestOnValidation()
{
	const std::vector<MNIST*>& test_data = App->dataset->GetValidationSet();
	int correct_answers = 0;
	float cost = 0;

	for (std::vector<MNIST*>::const_iterator it = test_data.begin(); it != test_data.end(); it++)
	{
		// Compute
		Eigen::MatrixXd input = (*it)->GetImage();
		int result = ComputeResult(input);

		int label = (*it)->GetLabel();

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(10,1));
		desired(label,1) = 1.0;

		// Calculate cost
		cost += Cost(input, desired);

		if (result == label)
		{
			correct_answers++;
		}
	}
	App->analitycs->AddResultValidation((float)correct_answers / (float) test_data.size(), cost);
}

void NeuralNetwork::TestOnTest()
{
	const std::vector<MNIST*>& test_data = App->dataset->GetTestSet();
	int correct_answers = 0;
	float cost = 0;

	for (std::vector<MNIST*>::const_iterator it = test_data.begin(); it != test_data.end(); it++)
	{
		// Compute
		Eigen::MatrixXd input = (*it)->GetImage();
		int result = ComputeResult(input);

		int label = (*it)->GetLabel();

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(10, 1));
		desired(label, 1) = 1.0;

		// Calculate cost
		cost += Cost(input, desired);

		if (result == label)
		{
			correct_answers++;
		}
	}
	App->analitycs->AddResultTest((float)correct_answers / (float)test_data.size(), cost);
}

void NeuralNetwork::TestOnTraining()
{
	const std::vector<MNIST*>& test_data = App->dataset->GetTrainingSet();
	int correct_answers = 0;
	float cost = 0;

	int start_pos = std::rand() % test_data.size() - 10000;

	if (start_pos < 0)
	{
		start_pos = 0;
	}

	for (int i = 0; i < 10000; i++)
	{
		const MNIST& image = *test_data[start_pos + i];

		// Compute
		Eigen::MatrixXd input = image.GetImage();
		int result = ComputeResult(input);

		int label = image.GetLabel();

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(10, 1));
		desired(label, 1) = 1.0;

		// Calculate cost
		cost += Cost(input, desired);

		if (result == label)
		{
			correct_answers++;
		}
	}
	App->analitycs->AddResultTraining((float)correct_answers / 10000.0f, cost);
}

float NeuralNetwork::Cost(const Eigen::MatrixXd& output, const Eigen::MatrixXd& desired)
{
	switch (cost_function)
	{
	case CF_QUADRATIC:
		return (output - desired).norm() * (output - desired).norm();

	case CF_CROSS_ENTHROPY:
	{
		float cost = 0.0000f;
		for (int i = 0; i < output.rows(); i++)
		{
			cost += -desired(i) * log(output(i)) - (1 - desired(i))* log(1 - output(i));
		}
		return cost;
	}

	default:
		std::cerr << "NeuralNetwork - TotalCost - No or unknown cost function" << std::endl;
		return 0.0f;
	}
}

int NeuralNetwork::GetResult(const Eigen::MatrixXd& output) const
{
	if (output.size() != 10)
	{
		std::cerr << "NeuralNetwork - GetResult - Invalid output" << std::endl;
		return -1;
	}

	int result = -1;
	double max_activation = 0.0;

	for (int i = 0; i < output.rows(); i++)
	{
		if (output(i,1) > max_activation)
		{
			result = i;
			max_activation = output(i, 1);
		}
	}

	if (result == -1)
	{
		std::cerr << "NeuralNetwork - GetResult - Invalid result" << std::endl;
	}

	return result;
}
