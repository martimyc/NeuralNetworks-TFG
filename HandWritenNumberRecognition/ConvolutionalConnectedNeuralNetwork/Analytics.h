#ifndef ANALYTICS
#define ANALYTICS

#include <vector>
#include <mutex>

#include "Module.h"

class Analitycs : public Module
{
public:
	Analitycs();
	~Analitycs();

	// Functionality
	bool Init() override;
	bool PreUpdate() override;
	bool Update() override;
	bool PostUpdate() override;
	bool CleanUp() override;

	void AddResultValidation(float ac, float c);
	void AddResultTest(float ac, float c);
	void AddResultTraining(float ac, float c);

private:
	std::mutex mtx;

	// Validation Results
	std::vector<float> validation_acuracy;
	std::vector<float> validation_cost;
	float validation_max_acuracy;	
	float validation_min_cost;
	int validation_max_acuracy_training_sesion;
	int validation_min_cost_training_sesion;

	// Test Results
	std::vector<float> test_acuracy;
	std::vector<float> test_cost;
	float test_max_acuracy;
	float test_min_cost;
	int test_max_acuracy_training_sesion;
	int test_min_cost_training_sesion;

	// Training Results
	std::vector<float> training_acuracy;
	std::vector<float> training_cost;
	float training_max_acuracy;
	float training_min_cost;
	int training_max_acuracy_training_sesion;
	int training_min_cost_training_sesion;
};

#endif //!ANALYTICS
