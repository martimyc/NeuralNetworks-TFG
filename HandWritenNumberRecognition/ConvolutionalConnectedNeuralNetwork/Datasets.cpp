#include "Datasets.h"

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <sstream>

#include "imgui.h"

#include "Globals.h"
#include "MNIST.h"

Dataset::Dataset(): Module("Dataset"), training_load_thread(nullptr), test_load_thread(nullptr)
{}

Dataset::~Dataset()
{}

bool Dataset::Init()
{
	training_load_thread = new std::thread(&Dataset::LoadTraining, std::ref(*this));
	test_load_thread = new std::thread(&Dataset::LoadTest, std::ref(*this));

	return true;
}

bool Dataset::PreUpdate()
{
	return true;
}

bool Dataset::Update()
{
	ImGui::Begin("Dataset");
	ImGui::Text("Number loaded training images: %i", training_set.size());
	ImGui::Text("Number loaded test images: %i", test_set.size());
	ImGui::End();

	/*if (training_load_state == LS_COMPLETED_SUCCESFULY && test_load_state == LS_COMPLETED_SUCCESFULY)
	{
		ImGui::Begin("Dataset Test");

		ImGui::Image((ImTextureID)test_set.front()->GetTexture(), ImVec2(200,200));
		ImGui::Text("%i",test_set.front()->GetLabel());
		std::stringstream ss;
		Eigen::IOFormat fmt;
		Eigen::MatrixXd image = test_set.front()->GetImage();
		for (int i = 0; i < image.rows(); i++)
		{
			for (int j = 0; j < image.cols(); j++)
			{
				if (image(i, j) > 0.00)
				{
					image(i, j) = 1;
				}
			}
		}
		ss << image.format(fmt);
		ImGui::Text(ss.str().c_str());

		ImGui::Image((ImTextureID)test_set.back()->GetTexture(), ImVec2(200, 200));
		ImGui::Text("%i", test_set.back()->GetLabel());
		std::stringstream ss2;
		const Eigen::MatrixXd& image = test_set.back()->GetImage();
		ss2 << image.format(fmt);
		ImGui::Text(ss2.str().c_str());

		ImGui::Image((ImTextureID)training_set.front()->GetTexture(), ImVec2(200, 200));
		ImGui::Text("%i", training_set.front()->GetLabel());
		std::stringstream ss3;
		ss3<< training_set.front()->GetImage().format(fmt);
		ImGui::Text(ss3.str().c_str());

		ImGui::Image((ImTextureID)training_set.back()->GetTexture(), ImVec2(200, 200));
		ImGui::Text("%i", training_set.back()->GetLabel());
		std::stringstream ss4;
		ss4 << training_set.back()->GetImage().format(fmt);
		ImGui::Text(ss4.str().c_str());

		ImGui::End();
	}*/

	return true;
}

bool Dataset::PostUpdate()
{
	if (training_load_state == LS_COMPLETED_WITH_ERRORS)
	{
		std::cerr << "Dataset - Training dataset load failed" << std::endl;
		return false;
	}
	else if (training_load_state == LS_COMPLETED_SUCCESFULY && training_load_thread != nullptr)
	{
		training_load_thread->join();
		delete training_load_thread;
		training_load_thread = nullptr;
	}

	if (test_load_state == LS_COMPLETED_WITH_ERRORS)
	{
		std::cerr << "Dataset - Test dataset load failed" << std::endl;
		return false;
	}
	else if (test_load_state == LS_COMPLETED_SUCCESFULY && test_load_thread != nullptr)
	{
		test_load_thread->join();
		delete test_load_thread;
		test_load_thread = nullptr;
	}

	return true;
}

bool Dataset::CleanUp()
{
	for (std::vector<MNIST*>::iterator it = training_set.begin(); it != training_set.end(); it++)
	{
		delete (*it);
	}

	for (std::vector<MNIST*>::iterator it = test_set.begin(); it != test_set.end(); it++)
	{
		delete (*it);
	}

	// In case loading is not finished
	if (training_load_thread != nullptr)
	{
		training_load_thread->join();
		delete training_load_thread;
		training_load_thread = nullptr;
	}

	if (test_load_thread != nullptr)
	{
		test_load_thread->join();
		delete test_load_thread;
		test_load_thread = nullptr;
	}

	return true;
}

void Dataset::LoadTraining()
{
	//Open files
	std::ifstream training_images("DataSets/train-images.idx3-ubyte", std::ifstream::binary);
	std::ifstream training_labels("DataSets/train-labels.idx1-ubyte");

	if (!training_images.is_open())
	{
		char error[255];
		strerror_s(error, errno);
		std::cerr << "Dataset - LoadTraining - Training images file can not be read - Error: " << error << std::endl;
		training_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	if (!training_labels.is_open())
	{
		char error[255];
		strerror_s(error, errno);
		std::cerr << "Dataset - LoadTraining - Training labels file can not be read - Error: " << error << std::endl;
		training_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	//Headers
	// Image file
	uint32_t image_magic;
	uint32_t num_images;
	uint32_t image_width;
	uint32_t image_height;

	// Label file
	uint32_t label_magic;
	uint32_t num_labels;

	unsigned char* header_bytes = new unsigned char[16];

	// Image file header
	if (training_images.read((char*)header_bytes, 16))
	{
		image_magic = ConvertToLittleEndian(header_bytes);
		num_images = ConvertToLittleEndian(&header_bytes[4]);
		image_width = ConvertToLittleEndian(&header_bytes[8]);
		image_height = ConvertToLittleEndian(&header_bytes[12]);
	}
	else
	{
		std::cerr << "Dataset - LoadTraining - Could not read image file header - Error: ";

		if (!training_images.goodbit)
		{
			if (training_images.eofbit)
				std::cerr << "End-Of-File reached while performing an extracting operation on an input stream. ";
			if (training_images.failbit)
				std::cerr << "The last input operation failed because of an error related to the internal logic of the operation itself. ";
			if (training_images.badbit)
				std::cerr << "Error due to the failure of an input/output operation on the stream buffer. ";
		}

		std::cerr << std::endl;

		training_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	// Label file header	
	if (training_labels.read((char*)header_bytes, 8))
	{
		label_magic = ConvertToLittleEndian(header_bytes);
		num_labels = ConvertToLittleEndian(&header_bytes[4]);
	}
	else
	{
		std::cerr << "Dataset - LoadTraining - Could not read label file header - Error: ";

		if (!training_images.goodbit)
		{
			if (training_images.eofbit)
				std::cerr << "End-Of-File reached while performing an extracting operation on an input stream. ";
			if (training_images.failbit)
				std::cerr << "The last input operation failed because of an error related to the internal logic of the operation itself. ";
			if (training_images.badbit)
				std::cerr << "Error due to the failure of an input/output operation on the stream buffer. ";
		}

		std::cerr << std::endl;

		training_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	if (num_images != num_labels)
	{
		std::cerr << "Dataset - LoadTraining - Unequal number of labels and images" << std::endl;
		training_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	// Images
	int num_training_images = num_images * TRAINING_PERCENTAGE;
	int num_validation_images = num_images * VALIDATION_PERCENTAGE;

	// Training	
	training_set.reserve(num_training_images);

	for (int i = 0; i < num_training_images; i++)
	{
		MNIST* new_image = new MNIST();
		new_image->LoadImage(training_images, image_width, image_height);
		training_set.push_back(new_image);
	}

	// Validation
	validation_set.reserve(num_validation_images);

	for (int i = 0; i < num_validation_images; i++)
	{
		MNIST* new_image = new MNIST();
		new_image->LoadImage(training_images, image_width, image_height);
		validation_set.push_back(new_image);
	}

	// Labels
	unsigned char label_byte;

	// Training
	for (int i = 0; i < num_training_images; i++)
	{
		training_set[i]->LoadLabel(training_labels);
	}

	// Validation
	for (int i = 0; i < num_validation_images; i++)
	{
		validation_set[i]->LoadLabel(training_labels);
	}
	
	training_load_state = LS_COMPLETED_SUCCESFULY;
}

void Dataset::LoadTest()
{
	//Open files
	std::ifstream test_images("DataSets/t10k-images.idx3-ubyte", std::ifstream::binary);
	std::ifstream test_labels("DataSets/t10k-labels.idx1-ubyte", std::ifstream::binary);

	if (!test_images.is_open())
	{
		char error[255];
		strerror_s(error, errno);
		std::cerr << "Dataset - LoadTest - Training images file can not be read - Error: " << error << std::endl;
		test_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	if (!test_labels.is_open())
	{
		char error[255];
		strerror_s(error, errno);
		std::cerr << "Dataset - LoadTest - Training labels file can not be read - Error: " << error << std::endl;
		test_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	//Headers
	// Image file
	uint32_t image_magic;
	uint32_t num_images;
	uint32_t image_width;
	uint32_t image_height;

	// Label file
	uint32_t label_magic;
	uint32_t num_labels;

	unsigned char* header_bytes = new unsigned char[16];

	// Image file header
	if (test_images.read((char*)header_bytes, 16))
	{
		image_magic = ConvertToLittleEndian(header_bytes);
		num_images = ConvertToLittleEndian(&header_bytes[4]);
		image_width = ConvertToLittleEndian(&header_bytes[8]);
		image_height = ConvertToLittleEndian(&header_bytes[12]);
	}
	else
	{
		std::cerr << "Dataset - LoadTest - Could not read image file header - Error: ";

		if (!test_images.goodbit)
		{
			if (test_images.eofbit)
				std::cerr << "End-Of-File reached while performing an extracting operation on an input stream. ";
			if (test_images.failbit)
				std::cerr << "The last input operation failed because of an error related to the internal logic of the operation itself. ";
			if (test_images.badbit)
				std::cerr << "Error due to the failure of an input/output operation on the stream buffer. ";
		}

		std::cerr << std::endl;

		test_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	// Label file header

	if (test_labels.read((char*)header_bytes, 8))
	{
		label_magic = ConvertToLittleEndian(header_bytes);
		num_labels = ConvertToLittleEndian(&header_bytes[4]);
	}
	else
	{
		std::cerr << "Dataset - LoadTest - Could not read label file header - Error: ";

		if (!test_images.goodbit)
		{
			if (test_images.eofbit)
				std::cerr << "End-Of-File reached while performing an extracting operation on an input stream. ";
			if (test_images.failbit)
				std::cerr << "The last input operation failed because of an error related to the internal logic of the operation itself. ";
			if (test_images.badbit)
				std::cerr << "Error due to the failure of an input/output operation on the stream buffer. ";
		}

		std::cerr << std::endl;

		test_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	if (num_images != num_labels)
	{
		std::cerr << "Dataset - LoadTest - Unequal number of labels and images" << std::endl;
		test_load_state = LS_COMPLETED_WITH_ERRORS;
		return;
	}

	// Images
	test_set.reserve(num_images);

	for (int i = 0; i < num_images; i++)
	{
		MNIST* new_image = new MNIST();
		new_image->LoadImage(test_images, image_width, image_height);
		test_set.push_back(new_image);
	}

	// Labels
	unsigned char label_byte;

	for (int i = 0; i < num_labels; i++)
	{
		test_set[i]->LoadLabel(test_labels);
	}

	test_load_state = LS_COMPLETED_SUCCESFULY;
}

void Dataset::GenTextures()
{
}

uint32_t Dataset::ConvertToLittleEndian(unsigned char * bytes)
{
	return uint32_t((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}
