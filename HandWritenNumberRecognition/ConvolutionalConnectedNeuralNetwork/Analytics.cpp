#include "Analytics.h"

#include "imgui.h"

Analitycs::Analitycs(): Module("Analytics")
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
	ImGui::Begin("Analytics");
	ImGui::Columns(3);

	// Validation Results
	if (validation_acuracy.size() > 0)
	{
		ImGui::Text("Validation Results:");

		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &validation_acuracy[0], validation_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &validation_cost[0], validation_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));
	}
	ImGui::NextColumn();

	// Test Results
	if (test_acuracy.size() > 0)
	{
		ImGui::Text("Test Results:");

		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &test_acuracy[0], test_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &test_cost[0], test_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));
	}
	ImGui::NextColumn();

	// Training Results
	if (training_acuracy.size() > 0)
	{
		ImGui::Text("Training Results:");

		ImGui::Text("Acuracy");
		ImGui::PlotHistogram("Results (%)", &training_acuracy[0], training_acuracy.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));

		ImGui::Text("Cost");
		ImGui::PlotHistogram("Results (%)", &training_cost[0], training_cost.size(), 0, 0, 3.402823466e+38F, 3.402823466e+38F, ImVec2(500, 200));
	}

	ImGui::End();

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
