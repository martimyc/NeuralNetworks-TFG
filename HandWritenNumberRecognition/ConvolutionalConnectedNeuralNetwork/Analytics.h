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

	inline void AddResultValidation(float ac, float c) { mtx.lock(); validation_acuracy.push_back(ac); validation_cost.push_back(c); mtx.unlock(); }
	inline void AddResultTest(float ac, float c) { mtx.lock(); test_acuracy.push_back(ac); test_cost.push_back(c);  mtx.unlock(); }
	inline void AddResultTraining(float ac, float c) { mtx.lock(); training_acuracy.push_back(ac); training_cost.push_back(c); mtx.unlock(); }

private:
	std::mutex mtx;

	// Validation Results
	std::vector<float> validation_acuracy;
	std::vector<float> validation_cost;

	// Test Results
	std::vector<float> test_acuracy;
	std::vector<float> test_cost;

	// Training Results
	std::vector<float> training_acuracy;
	std::vector<float> training_cost;
};

#endif //!ANALYTICS
