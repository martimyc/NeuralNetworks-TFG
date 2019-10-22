#ifndef DATASET
#define DATASET

#include <vector>
#include <thread>

#include "Module.h"

class MNIST;

enum LOAD_STATE
{
	LS_IN_PROGRESS = 0,
	LS_COMPLETED_SUCCESFULY,
	LS_COMPLETED_WITH_ERRORS
};

class Dataset : public Module
{
public:
	Dataset();
	~Dataset();

	// Functionality
	bool Init() override;
	bool PreUpdate() override;
	bool Update() override;
	bool PostUpdate() override;
	bool CleanUp() override;

	// Getters & Setters
	inline const MNIST& GetTrainingImage(int image) const { return *training_set[image]; }
	inline const std::vector<MNIST*>& GetTrainingSet() const { return training_set; }
	inline const std::vector<MNIST*>& GetTestSet() const { return test_set; }
	inline const std::vector<MNIST*>& GetValidationSet() const { return validation_set; }
	inline bool LoadingDone() const { return(training_load_state == LS_COMPLETED_SUCCESFULY && test_load_state == LS_COMPLETED_SUCCESFULY); }

private:
	// Load
	void LoadTraining();
	void LoadTest();
	void GenTextures();

	// Utility
	uint32_t ConvertToLittleEndian(unsigned char* bytes);

private:
	std::vector<MNIST*> training_set;
	std::vector<MNIST*> validation_set;
	std::vector<MNIST*> test_set;

	// Load thread
	std::thread* training_load_thread;
	LOAD_STATE training_load_state;

	std::thread* test_load_thread;
	LOAD_STATE test_load_state;
};


#endif // !DATASET
