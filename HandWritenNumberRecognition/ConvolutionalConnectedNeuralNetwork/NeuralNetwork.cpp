#include "NeuralNetwork.h"

#include <iostream>
#include <random>
#include <time.h>
#include <math.h>

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

void NeuralNetwork::FeedForward(Eigen::MatrixXd& input) const
{
	for (std::vector<Layer*>::const_iterator layer = layers.begin(); layer != layers.end(); layer++)
	{
		input = (*layer)->FeedForward(input);
	}
}

void NeuralNetwork::BackPropagation(const Eigen::MatrixXd& cost, float eta, int mini_batch_size, float lambda)
{
	Eigen::MatrixXd gradient(cost);

	for (std::vector<Layer*>::const_reverse_iterator layer = layers.rbegin(); layer != layers.rend(); layer++)
	{
		gradient = (*layer)->BackPropagate(gradient, eta, mini_batch_size, lambda);
	}
}

void NeuralNetwork::SGD(const std::vector<MNIST*>& training_data, int epochs, int mini_batch_size, float eta, float lambda)
{
	state = S_TRAINING;

	// todo paralel

	for (int i = 0; i < epochs; i++)
	{
		// Create random mini batch
		// Starting point for mini batch. Somewhere between 0 and training data size - mini batch size
		int start = std::rand() % (training_data.size() - mini_batch_size);
		std::vector<MNIST*>::const_iterator start_point = training_data.begin() + start;
		std::vector<MNIST*>::const_iterator end_point = training_data.begin() + start + mini_batch_size;

		std::vector<MNIST*> mini_batch(start_point, end_point);

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
	FeedForward(input);
	return GetResult(input);
}

// Layers
FullyConnectedLayer * NeuralNetwork::AddFullyConnectedLayer(int layer_neurons, int previous_layer_neurons, ACTIVATION_FUNCTION activation_funct, bool regularization)
{
	layers.push_back(new FullyConnectedLayer(layer_neurons, previous_layer_neurons, activation_funct, regularization));
	return (FullyConnectedLayer *)layers.back();
}

ConvolutionLayer * NeuralNetwork::AddConvolutionLayer(int k_size, POOLING pooling, ACTIVATION_FUNCTION activation_function, int num_filters, int input_image_size, bool regularization)
{
	layers.push_back(new ConvolutionLayer(k_size, pooling, activation_function, num_filters, input_image_size, regularization));
	return (ConvolutionLayer *)layers.back();
}

// UI
void NeuralNetwork::Info() const
{
	ImGui::Text("Layers: %i", layers.size());

	ImGui::Columns(layers.size(), "word-wrapping");

	for (std::vector<Layer*>::const_iterator it = layers.begin(); it != layers.end(); it++)
	{
		(*it)->UI();

		if (*it != layers.back())
		{
			ImGui::NextColumn();
		}
	}

	ImGui::Columns();
}

void NeuralNetwork::Debug()
{
	// Image
	Eigen::MatrixXd debug_image (28,28);
	debug_image <<
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

	// Feed forward
	Eigen::MatrixXd input(debug_image);
	DebugFeedForward(input);

	// Desired outcome
	Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(input.rows(), 1));
	desired(0, 0) = 1.0;

	Eigen::IOFormat fmt;
	//std::cerr << "Output:\n" << input.format(fmt) << "\nDesired:\n" << desired.format(fmt) << "\nCost:" << Delta(input, desired) << std::endl;
	std::cerr << "Output:\n" << input.format(fmt) << std::endl;

	// With the first layers error we can back propagate and get the next layer's error as well as change its weights and biases
	DebugBackPropagation(Delta(input, desired));
}

void NeuralNetwork::DebugFeedForward(Eigen::MatrixXd & input) const
{
	std::cerr << "Forward prop:\n" << std::endl;

	for (int i = 0; i < layers.size(); i++)
	{
		Eigen::IOFormat fmt;
		std::cerr << "Layer " << i << " Input:\n" << input.format(fmt) << std::endl;
		//std::cerr << "Layer " << i << " Weights:\n" << layers[i]->AsFullyConnected()->GetWeights().format(fmt) << std::endl;
		//std::cerr << "Layer " << i << " Biases:\n" << layers[i]->AsFullyConnected()->GetBiases().format(fmt) << std::endl;

		input = layers[i]->FeedForward(input);

		//std::cerr << "\nOnput:\n" << input.format(fmt) << std::endl;
	}
}

void NeuralNetwork::DebugBackPropagation(const Eigen::MatrixXd & cost)
{
	Eigen::MatrixXd gradient(cost);

	std::cerr << "Back prop:\n" << std::endl;

	for (int i = layers.size() - 1; i >= 0; i--)
	{
		Eigen::IOFormat fmt;
		std::cerr << "Layer " << i << " Input:\n" << gradient.format(fmt) << std::endl;

		gradient = layers[i]->BackPropagate(gradient, 1.0, 1.0, 1.0);

		//std::cerr << "\nOnput:\n" << gradient.format(fmt) << std::endl;
	}
}

void NeuralNetwork::DebugLayer()
{
	layers.front()->AsFullyConnected()->Debug();
}

void NeuralNetwork::DebugConvolution()
{
	layers.front()->AsConvolution()->Debug();
}

// Private
// Work
void NeuralNetwork::UpdateWithMiniBatch(std::vector<MNIST*>& mini_batch, float eta, float lambda)
{
	for (std::vector<MNIST*>::iterator it = mini_batch.begin(); it != mini_batch.end(); it++)
	{
		// Feed forward
		Eigen::MatrixXd input((*it)->GetImage());
		FeedForward(input);

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(input.rows(), 1));
		desired((*it)->GetLabel(), 0) = 1.0;

		// Back Propagate
		BackPropagation(Delta(input, desired), eta, mini_batch.size(), lambda);

		// Clean up layers for next image
		for (std::vector<Layer*>::const_iterator it = layers.begin(); it != layers.end(); it++)
		{
			(*it)->CleanUp();
		}
	}
}

// Cost Functions
Eigen::MatrixXd NeuralNetwork::Delta(const Eigen::MatrixXd & activation, const Eigen::MatrixXd & desired) const
{
	// Check
	switch (cost_function)
	{
	case CF_QUADRATIC: return activation - desired;
	case CF_CROSS_ENTHROPY: return activation - desired;
	case CF_LOG_LIKELIHOOD: 
	{
		Eigen::MatrixXd output = activation - desired;
		/*for (int i = 0; i < desired.rows(); i++)
		{
			output(i) = -log(1.0 - output(i));
		}*/
		return output;
	}

	default:
		std::cerr << "NeuralNetwork - Delta - Unknown cost function" << std::endl;
		return Eigen::MatrixXd();
	}
}

// Test
void NeuralNetwork::TestOnValidation()
{
	const std::vector<MNIST*>& validation_data = App->dataset->GetValidationSet();
	int correct_answers = 0;
	float cost = 0;

	for (std::vector<MNIST*>::const_iterator it = validation_data.begin(); it != validation_data.end(); it++)
	{
		// Compute
		Eigen::MatrixXd input = (*it)->GetImage();
		int result = ComputeResult(input);

		int label = (*it)->GetLabel();

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(10,1));
		desired(label,0) = 1.0;

		// Calculate cost
		cost += Cost(input, desired);

		if (result == label)
		{
			correct_answers++;
		}
	}
	App->analitycs->AddResultValidation((float)correct_answers / (float)validation_data.size(), cost);
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
		desired(label, 0) = 1.0;

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
	const std::vector<MNIST*>& training_data = App->dataset->GetTrainingSet();
	int correct_answers = 0;
	float cost = 0;

	int start_pos = std::rand() % training_data.size() - 10000;

	if (start_pos < 0)
	{
		start_pos = 0;
	}

	for (int i = 0; i < 10000; i++)
	{
		const MNIST& image = *training_data[start_pos + i];

		// Compute
		Eigen::MatrixXd input = image.GetImage();
		int result = ComputeResult(input);

		int label = image.GetLabel();

		// Desired outcome
		Eigen::MatrixXd desired(Eigen::MatrixXd::Zero(10, 1));
		desired(label, 0) = 1.0;

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
	case CF_LOG_LIKELIHOOD:
		return -log(output(desired.maxCoeff(), 0));

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
		if (output(i,0) > max_activation)
		{
			result = i;
			max_activation = output(i, 0);
		}
	}

	if (result == -1)
	{
		std::cerr << "NeuralNetwork - GetResult - Invalid result" << std::endl;
	}

	return result;
}
