#ifndef MNIST_
#define MNIST_

#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include "GL/glew.h"

class MNIST
{
public:
	MNIST();
	~MNIST();

	// Setters & Getters
	const GLuint& GetTexture();
	inline const Eigen::MatrixXd& GetImage() const { return pixels; }
	inline int GetLabel() const { return label; }

	// Load
	void LoadImage(std::ifstream& file, int width, int height);
	void LoadLabel(std::ifstream& file);

private:
	void InitTexture();

private:
	Eigen::MatrixXd pixels;
	int label;
	int width;
	int height;
	GLuint texture;
};

#endif // !MNIST_

