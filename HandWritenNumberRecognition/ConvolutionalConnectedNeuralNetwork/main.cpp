#include <vector>
#include <iostream>

#include "SDL.h"

#include "Application.h"
#include "NeuralNetwork.h"

Application* App = nullptr;

int main(int argc, char * argv[])
{
	App = new Application("Hand Writen Recognition Algorithm");

	// Init
	if (App->Init())
	{
		std::cerr << "All modules initialized succesfully\n" << std::endl;
	}
	else
	{
		std::cerr << "App initialization failed\n" << std::endl;
		return 0;
	}

	// Start
	if (App->Start())
	{
		std::cerr << "App start up completed correctly\n" << std::endl;
	}
	else
	{
		std::cerr << "App start up failed\n" << std::endl;
		return 0;
	}

	// Update
	while (App->IsRunning())
	{
		if (!App->Update())
		{
			std::cerr << "App update failed\n" << std::endl;
			return 0;
		}
	}

	// Clean up
	if (App->CleanUp())
	{
		std::cerr << "App clean up completed correctly\n" << std::endl;
	}
	else
	{
		std::cerr << "App clean up failed\n" << std::endl;
		return 0;
	}

	delete App;
	App = nullptr;

	/*SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, NULL);

	// Background color
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

	// Clear Screen
	SDL_RenderClear(renderer);


	NeuralNetwork nn;

	int input = 784;

	std::vector<int> hidden_layers;
	hidden_layers.push_back(15);

	int output = 10;

	if (!nn.Initialize(input, hidden_layers, output))
		std::cout << "Main - Neural Network initialized incorrectly" << std::endl;*/

	return 0;
}