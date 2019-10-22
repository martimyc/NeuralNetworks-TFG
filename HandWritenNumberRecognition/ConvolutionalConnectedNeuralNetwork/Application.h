#ifndef APP
#define APP

#include <vector>
#include <string>

// Modules
class Module;
class UI;
class Window;
class Dataset;
class ML;
class Analitycs;

class Application
{
public:
	Application(const std::string& name);
	~Application();

	// Functionality
	bool Init();
	bool Start();
	bool Update();
	bool CleanUp();

	// Getters & Setters
	inline bool IsRunning() const { return runing; }
	inline void Quit() { runing = false; }

public:
	// Modules
	Window* window;
	Dataset* dataset;
	ML* ml;
	Analitycs* analitycs;
	UI* ui;

private:
	std::vector<Module*> modules;
	std::string name;
	bool runing;
};

extern Application* App;

#endif //!APP