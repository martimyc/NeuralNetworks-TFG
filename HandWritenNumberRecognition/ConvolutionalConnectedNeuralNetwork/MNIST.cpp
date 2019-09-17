#include "MNIST.h"

MNIST::MNIST()
{}

MNIST::~MNIST()
{}

const GLuint & MNIST::GetTexture()
{
	if (texture == 0)
	{
		InitTexture();
	}

	return texture;
}

void MNIST::LoadImage(std::ifstream& file, int width, int height)
{
	unsigned char byte;

	this->width = width;
	this->height = height;

	pixels.resize(width * height);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			file.read((char*)&byte, 1);
			pixels[i*width + j] = float((int)byte) / 255.0;
		}
	}
}

void MNIST::LoadLabel(std::ifstream& file)
{
	unsigned char byte;
	file.read((char*)&byte, 1);
	label = (int)byte;
}

void MNIST::InitTexture()
{
	glGenTextures(1, &texture);

	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	float color[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenerateMipmap(GL_TEXTURE_2D);

	float* texture_pixels = new float[width * height * 3];

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int num_pixel = i * height * 3 + j * 3;
			float pixel_val = pixels[i * width + j];

			texture_pixels[num_pixel] = pixel_val;
			texture_pixels[num_pixel + 1] = pixel_val;
			texture_pixels[num_pixel + 2] = pixel_val;
		}
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, texture_pixels);
}