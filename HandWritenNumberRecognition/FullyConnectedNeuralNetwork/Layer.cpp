#include "..\ConvolutionalConnectedNeuralNetwork\Layer.h"
#include "Layer.h"

#include <iostream>

// Random
std::random_device Layer::rand_device;
std::mt19937 Layer::engine(rand_device());

Layer::Layer(LAYER_TYPE type): type(type)
{}

Layer::~Layer()
{}

Eigen::VectorXd Layer::ActivationFunction(const Eigen::VectorXd& vec) const
{
	switch (activation_function)
	{
	case AF_SIGMOID: return vec.unaryExpr(&Sigmoid);
	default:
		std::cerr << "Layer - ActivationFunction - unknown activation function" << std::endl;
		return Eigen::VectorXd();
	}
}

Eigen::MatrixXd Layer::ActivationFunction(const Eigen::MatrixXd & mat) const
{
	switch (activation_function)
	{
	case AF_SIGMOID: return mat.unaryExpr(&Sigmoid);
	default:
		std::cerr << "Layer - ActivationFunction - unknown activation function" << std::endl;
		return Eigen::MatrixXd();
	}
}

double Layer::Sigmoid(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

double Layer::SigmoidPrime(double z)
{
	return Sigmoid(z)*(1.0 - Sigmoid(z));
}

FullyConnectedLayer * Layer::AsFullyConnected()
{
	if (type != LT_FULLY_CONNECTED)
	{
		std::cerr << "Layer - AsFullyConnected - Layer is not fully connected layer" << std::endl;
		return nullptr;
	}
	
	return (FullyConnectedLayer*)this;	
}