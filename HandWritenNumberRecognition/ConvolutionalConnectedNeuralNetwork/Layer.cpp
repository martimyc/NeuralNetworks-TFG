#include "Layer.h"

#include <iostream>

// Random
std::random_device Layer::rand_device;
std::mt19937 Layer::engine(rand_device());

Layer::Layer(LAYER_TYPE type, bool regularization) : type(type), regularization(regularization)
{}

Layer::~Layer()
{}

FullyConnectedLayer * Layer::AsFullyConnected()
{
	if (type != LT_FULLY_CONNECTED)
	{
		std::cerr << "Layer - AsFullyConnected - Layer is not fully connected layer" << std::endl;
		return nullptr;
	}

	return (FullyConnectedLayer*)this;
}

ConvolutionLayer * Layer::AsConvolution()
{
	if (type != LT_CONVOLUTION)
	{
		std::cerr << "Layer - AsConvolution - Layer is not convolution layer" << std::endl;
		return nullptr;
	}

	return (ConvolutionLayer*)this;
}