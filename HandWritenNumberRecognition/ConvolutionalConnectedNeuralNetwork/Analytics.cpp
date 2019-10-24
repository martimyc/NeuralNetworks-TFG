#include "Analytics.h"

#include <limits>

#include "imgui.h"
#include "Globals.h"

Analitycs::Analitycs():
	Module("Analytics"),
	validation_max_acuracy(0.0f),
	validation_min_cost(std::numeric_limits<float>::max()),
	validation_max_acuracy_training_sesion(0),
	validation_min_cost_training_sesion(0),
	test_max_acuracy(0.0f),
	test_min_cost(std::numeric_limits<float>::max()),
	test_max_acuracy_training_sesion(0),
	test_min_cost_training_sesion(0),
	training_max_acuracy(0.0f),
	training_min_cost(std::numeric_limits<float>::max()),
	training_max_acuracy_training_sesion(0),
	training_min_cost_training_sesion(0)
{}

Analitycs::~Analitycs()
{}

bool Analitycs::Init()
{
	return true;
}

bool Analitycs::PreUpdate()
{
	return true;
}

bool Analitycs::Update()
{
	// Validation Results
	if (validation_acuracy.size() > 0)
	{
		ImGui::Begin("Validation Results:");

		ImGui::Text("Best acuracy: %f at training session %i", validation_max_acuracy, validation_max_acuracy_training_sesion);
		ImGui::Text("Lowest cost: %f at training session %i", validation_min_cost, validation_min_cost_training_sesion);
		ImGui::Separator();

		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &validation_acuracy[0], validation_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &validation_cost[0], validation_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);

		ImGui::End();
	}

	// Test Results
	if (test_acuracy.size() > 0)
	{
		ImGui::Begin("Test Results:");

		ImGui::Text("Best acuracy: %f at training session %i after %i seconds of training", test_max_acuracy, test_max_acuracy_training_sesion, elapsed_time);
		ImGui::Text("Lowest cost: %f at training session %i", test_min_cost, test_min_cost_training_sesion);
		ImGui::Separator();

		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &test_acuracy[0], test_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &test_cost[0], test_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);
		
		ImGui::End();
	}

	// Training Results
	if (training_acuracy.size() > 0)
	{
		ImGui::Begin("Training Results:");

		ImGui::Text("Best acuracy: %f at training session %i", training_max_acuracy, training_max_acuracy_training_sesion);
		ImGui::Text("Lowest cost: %f at training session %i", training_min_cost, training_min_cost_training_sesion);
		ImGui::Separator();
		
		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &training_acuracy[0], training_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &training_cost[0], training_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, HISTOGRAM_SIZE);
		
		ImGui::End();
	}

	return true;
}

bool Analitycs::PostUpdate()
{
	return true;
}

bool Analitycs::CleanUp()
{
	return true;
}

void Analitycs::AddResultValidation(float ac, float c)
{
	mtx.lock();
	if (validation_max_acuracy < ac)
	{
		validation_max_acuracy = ac;
		validation_max_acuracy_training_sesion = validation_acuracy.size();
	}

	if (validation_min_cost > c)
	{
		validation_min_cost = c;
		validation_min_cost_training_sesion = validation_acuracy.size();
	}

	validation_acuracy.push_back(ac);
	validation_cost.push_back(c);
	mtx.unlock();
}

void Analitycs::AddResultTest(float ac, float c)
{
	mtx.lock();
	if (test_max_acuracy < ac)
	{
		test_max_acuracy = ac;
		test_max_acuracy_training_sesion = test_acuracy.size();
		time(&elapsed_time);
		elapsed_time = elapsed_time - start_time;
	}

	if (test_min_cost > c)
	{
		test_min_cost = c;
		test_min_cost_training_sesion = test_acuracy.size();
	}

	test_acuracy.push_back(ac);
	test_cost.push_back(c);
	mtx.unlock();
}

void Analitycs::AddResultTraining(float ac, float c)
{
	mtx.lock();
	if (training_max_acuracy < ac)
	{
		training_max_acuracy = ac;
		training_max_acuracy_training_sesion = training_acuracy.size();
	}

	if (training_min_cost > c)
	{
		training_min_cost = c;
		training_min_cost_training_sesion = training_acuracy.size();
	}

	training_acuracy.push_back(ac);
	training_cost.push_back(c);
	mtx.unlock();
}

void Analitycs::StartTimer()
{
	time(&start_time);
}
