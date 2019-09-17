#ifndef WINDOW
#define WINDOW

#include <string>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl.h"

#include "Module.h"

class SDL_Window;
class SDL_Renderer;

class Window : public Module
{
public:
	Window(const std::string& title, int width, int height);
	~Window();

	// Functionality
	bool Init() override;
	bool PreUpdate() override;
	bool Update() override;
	bool PostUpdate() override;
	bool CleanUp() override;

private:
	// Events
	void PollEvents();
	void ClearScreen() const;

	// Getters & Setters

private:
	std::string title;
	int width;
	int height;

	// SDL
	SDL_Window* window;
	SDL_Renderer* renderer;

	// ImGUI
	ImGuiContext* context;
	ImGuiIO* io;
	ImVec4 clear_color;
};

#endif //!WINDOW