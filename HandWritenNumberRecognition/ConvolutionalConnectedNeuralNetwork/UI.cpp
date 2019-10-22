#include "UI.h"

#include "imgui.h"

UI::UI(): Module("UI")
{}

UI::~UI()
{}

bool UI::Init()
{
	return true;
}

bool UI::Start()
{
	return true;
}

bool UI::PreUpdate()
{
	return true;
}

bool UI::Update()
{
	ImGui::Begin("Used external libraries & resources");

	ImGui::Text("Eigen 3.3.7");
	ImGui::Text("ImGUI 1.72b");
	ImGui::Text("Glew 2.1.0");
	ImGui::Text("SDL2 2.0.10");
	ImGui::Text("OpenGL 3.0");
	ImGui::Text("glsl version 130");
	ImGui::Text("MNIST dataset");
	ImGui::Separator();
	ImGui::TextWrapped("Many thanks to all of these, without them this project would not have been posible.");

	ImGui::End();

	ImGui::Begin("Bibliography");

	ImGui::TextWrapped("Neural Networks and Deep Learning\nby Michael Nielsen\nJun 2019\nhttp://neuralnetworksanddeeplearning.com/");
	ImGui::Separator();
	ImGui::TextWrapped("Backpropagation In Convolutional Neural Networks\nby Jefkine\n5 September 2016\nhttps://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/");
	ImGui::Separator();
	ImGui::TextWrapped("ml-cheatsheet.readthedocs.io");
	ImGui::Separator();
	ImGui::TextWrapped("Deep learning youtube series\nby 3Blue1Brown\nOctober 5, 2017\nhttps://www.youtube.com/watch?v=aircAruvnKk");
	ImGui::Separator();
	ImGui::TextWrapped("Many thanks to all of these, without them this project would not have been posible.");

	ImGui::End();

	return true;
}

bool UI::PostUpdate()
{
	return true;
}

bool UI::CleanUp()
{
	return true;
}
