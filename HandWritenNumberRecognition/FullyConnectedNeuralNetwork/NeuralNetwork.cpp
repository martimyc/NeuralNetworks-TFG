#include "NeuralNetwork.h"

#include <iostream>
#include <random>
#include <time.h>

#include "imgui.h"

#include "Analytics.h"
#include "Application.h"
#include "Datasets.h"
#include "MNIST.h"
#include "FullyConnectedLayer.h"

NeuralNetwork::NeuralNetwork(int input, const std::vector<int>& hidden, int output, COST_FUNCTION cost_function, bool regularization): num_inputs(input), state(S_READY), cost_function(cost_function), regularization(regularization)
{
	if (input <= 0)
	{
		std::cout << "Neural Network - Initialize - Invalid number of input neurons" << std::endl;
		return;
	}

	if (output <= 0)
	{
		std::cout << "Neural Network - Initialize - Invalid number of output neurons" << std::endl;
		return;
	}

	for (int i = 0; i < hidden.size(); i++)
	{
		if (hidden[i] <= 0)
		{
			std::cout << "Neural Network - Initialize - Invalid number of hidden neurons in layer " << i << std::endl;
			return;
		}
	}

	layers.reserve(hidden.size() + output);

	for (int i = 0; i < hidden.size(); i++)
	{
		if (layers.size() == 0)
		{
			layers.push_back(new FullyConnectedLayer(hidden[i], input));
		}
		else
		{
			layers.push_back(new FullyConnectedLayer(hidden[i], hidden[i - 1]));
		}
	}

	layers.push_back(new FullyConnectedLayer(output, hidden.back()));

	// Seed rand for later
	std::srand(time(NULL));
}

NeuralNetwork::~NeuralNetwork()
{
	for (std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
	{
		delete *it;
	}
}

const Eigen::VectorXd& NeuralNetwork::FeedForward(const Eigen::VectorXd& input)
{
	if (num_inputs != input.size())
	{
		std::cerr << "NeuralNetwork - FeedForward - Invalid input" << std::endl;
		return Eigen::VectorXd(); // will go out of scope
	}

	// Feed forward every layer /w the last layer's activations (input for the first layer)
	const Eigen::VectorXd* last_layer_activation = &layers.front()->FeedForward(input);

	for (std::vector<Layer*>::iterator layer = layers.begin() + 1; layer != layers.end(); layer++)
	{
		last_layer_activation = &(*layer)->FeedForward(*last_layer_activation);
	}

	return *last_layer_activation;
}

void NeuralNetwork::BackPropagation(const Eigen::VectorXd & error, const Eigen::VectorXd & input, float eta, int mini_batch_size, float lambda)
{
	/*
	Back propagate the error from each layer to the previous until we go from output to imput.
	We collect each layer's error to later use to update weights and biases
	*/

	std::vector<Eigen::VectorXd> layer_errors;
	layer_errors.reserve(layers.size());

	// Output layer's error is the overall error
	layer_errors.push_back(error);

	for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin() + 1; layer != layers.rend(); layer++)
	{
		layer_errors.push_back((*layer)->BackPropagate( layer_errors.back(), (*(layer - 1))->GetWeights()));
	}

	/*
	We update each layer's weights & biases after back propagating the error all the way,
	otherwise the weights used to compute a layer's error would have already been changed when
	back propagating the other layers closer to the output
	*/

	// No regularization 
	if (lambda < 0.00001f)
	{
		std::vector<Eigen::VectorXd>::iterator err = layer_errors.begin();
		std::vector<Layer*>::reverse_iterator current_layer = layers.rbegin();
		std::vector<Layer*>::reverse_iterator previous_layer = current_layer + 1;

		for (; previous_layer != layers.rend(); current_layer++, err++, previous_layer++)
		{
			(*current_layer)->UpdateWeightsAndBiases(*err, (*previous_layer)->GetActivation(), eta / mini_batch_size);
		}

		// Update first layer with input as activations
		layers.front()->UpdateWeightsAndBiases(layer_errors.back(), input, eta / mini_batch_size);
	}
	// L2 Regularization
	else
	{
		std::vector<Eigen::VectorXd>::iterator err = layer_errors.begin();
		std::vector<Layer*>::reverse_iterator current_layer = layers.rbegin();
		std::vector<Layer*>::reverse_iterator previous_layer = current_layer + 1;

		for (; previous_layer != layers.rend(); current_layer++, err++, previous_layer++)
		{
			(*current_layer)->UpdateWeightsAndBiasesRegular(*err, (*previous_layer)->GetActivation(), eta, mini_batch_size, lambda);
		}

		// Update first layer with input as activations
		layers.front()->UpdateWeightsAndBiasesRegular(layer_errors.back(), input, eta, mini_batch_size, lambda);
	}
}

void NeuralNetwork::SGD(const std::vector<MNIST*>& training_data, int epochs, int mini_batch_size, float eta, float lambda)
{
	state = S_TRAINING;

	for (int i = 0; i < epochs; i++)
	{
		// Create mini batch
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

void NeuralNetwork::UpdateWithMiniBatch(std::vector<MNIST*>& mini_batch, float eta, float lambda)
{
	for (std::vector<MNIST*>::iterator it = mini_batch.begin(); it != mini_batch.end(); it++)
	{
		// Feed forward
		const Eigen::VectorXd& image((*it)->GetImage());
		const Eigen::VectorXd& activation = FeedForward(image);

		// Desired outcome
		Eigen::VectorXd desired(Eigen::VectorXd::Zero(activation.rows()));
		desired[(*it)->GetLabel()] = 1.0;

		/*
		This first(last) layer's error is calculated by multiplying the cost derivative in regards to activation (the
		variation of cost in with regards to each activation cost) by the variation of the activation function in that neuron's
		Z (sigmoid prime of z)
		To compute this better we create a diagonal matrix with the values resulting from sigmoid prime of Z and multiply it by
		the cost derivative resulting vector
		*/

		Eigen::MatrixXd sigmoid_prime_z (layers.back()->GetZ().unaryExpr(&FullyConnectedLayer::SigmoidPrime).asDiagonal());

		Eigen::VectorXd delta = sigmoid_prime_z * Quadratic(activation, desired);

		// With the first layers error we can back propagate and get the next layer's error as well as change its weights and biases
		BackPropagation(delta, image, eta, mini_batch.size(), lambda);
	}
}

void NeuralNetwork::TestOnValidation()
{
	const std::vector<MNIST*>& test_data = App->dataset->GetValidationSet();
	int correct_answers = 0;
	float cost = 0;

	for (std::vector<MNIST*>::const_iterator it = test_data.begin(); it != test_data.end(); it++)
	{
		Eigen::VectorXd output = FeedForward((*it)->GetImage());
		int result = GetResult(output);

		int label = (*it)->GetLabel();

		Eigen::VectorXd desired(Eigen::VectorXd::Zero(10));
		desired(label) = 1.0;

		cost += Cost(output, desired);

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
		Eigen::VectorXd output = FeedForward((*it)->GetImage());
		int result = GetResult(output);

		int label = (*it)->GetLabel();

		Eigen::VectorXd desired(Eigen::VectorXd::Zero(10));
		desired(label) = 1.0;

		cost += Cost(output, desired);

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

		Eigen::VectorXd output = FeedForward(image.GetImage());
		int result = GetResult(output);

		int label = image.GetLabel();

		Eigen::VectorXd desired(Eigen::VectorXd::Zero(10));
		desired(label) = 1.0;

		cost += Cost(output, desired);

		if (result == label)
		{
			correct_answers++;
		}
	}
	App->analitycs->AddResultTraining((float)correct_answers / 10000.0f, cost);
}

float NeuralNetwork::Cost(const Eigen::VectorXd& output, const Eigen::VectorXd& desired)
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

int NeuralNetwork::GetResult(const Eigen::VectorXd& output) const
{
	if (output.size() != 10)
	{
		std::cerr << "NeuralNetwork - GetResult - Invalid output" << std::endl;
		return -1;
	}

	int result = -1;
	double max_activation = 0.0;

	for (int i = 0; i < output.size(); i++)
	{
		if (output[i] > max_activation)
		{
			result = i;
			max_activation = output[i];
		}
	}

	if (result == -1)
	{
		std::cerr << "NeuralNetwork - GetResult - Invalid result" << std::endl;
	}

	return result;
}

Layer & NeuralNetwork::GetLastLayer()
{
	return *layers[layers.size() - 1];
}

Eigen::VectorXd NeuralNetwork::Delta(const Eigen::VectorXd& activation, const Eigen::VectorXd& desired)
{
	switch (cost_function)
	{

	case CF_QUADRATIC:
	{
		Eigen::MatrixXd sigmoid_prime_z(layers.back()->GetZ().unaryExpr(&FullyConnectedLayer::SigmoidPrime).asDiagonal());
		return sigmoid_prime_z * Quadratic(activation, desired);
	}

	case CF_CROSS_ENTHROPY:
	{

	}

	default:
	{
		std::cerr << "NeuralNetwork - Delta - No or unknown cost function" << std::endl;
		return Eigen::VectorXd();
	}

	}
}

Eigen::VectorXd NeuralNetwork::Quadratic(const Eigen::VectorXd & activation, const Eigen::VectorXd & desired)
{
	return activation - desired;
}

Eigen::VectorXd NeuralNetwork::CrossEntropy(const Eigen::VectorXd & activation, const Eigen::VectorXd & desired)
{
	return Eigen::VectorXd();
}

void NeuralNetwork::Info() const
{
	ImGui::Text("Input neurons: %i", num_inputs);
	ImGui::Text("Hidden layers: %i", layers.size() - 1);

	for (int i = 0; i < layers.size() - 1; i++)
	{
		ImGui::Text("Hidden layer %i neurons: %i", i, layers[i]->GetNumNeurons());
	}

	ImGui::Text("Output neurons: %i", layers.back()->GetNumNeurons());
}
