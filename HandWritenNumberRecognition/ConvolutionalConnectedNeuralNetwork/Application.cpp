#include "Application.h"

#include <iostream>

#include "Globals.h"

#include "Module.h"
#include "Window.h"
#include "Datasets.h"
#include "ML.h"
#include "Analytics.h"
#include "UI.h"

Application::Application(const std::string& name): name (name), runing(true), window(nullptr), dataset(nullptr), ml(nullptr), analitycs(nullptr), ui(nullptr)
{
	// Create modules
	window = new Window(name, SCREEN_WIDTH, SCREEN_HEIGHT);
	dataset = new Dataset();
	ml = new ML();
	analitycs = new Analitycs();
	ui = new UI();

	// Add modules
	modules.push_back(window);
	modules.push_back(dataset);
	modules.push_back(ml);
	modules.push_back(analitycs);
	modules.push_back(ui);
}

Application::~Application()
{
	for (std::vector<Module*>::reverse_iterator it = modules.rbegin(); it != modules.rend(); it++)
	{
		delete *it;
	}
}

bool Application::Init()
{
	for (std::vector<Module*>::iterator it = modules.begin(); it != modules.end(); it++)
	{
		if ((*it)->Init())
		{
			(*it)->SetActive(true);
		}
		else
		{
			std::cerr << "Application - Init - Module " << (*it)->GetName() << " could not be initialized" << std::endl;
			return false;
		}
	}
	return true;
}

bool Application::Start()
{
	for (std::vector<Module*>::iterator it = modules.begin(); it != modules.end(); it++)
	{
		if ((*it)->Active())
		{
			if (!(*it)->Start())
			{
				std::cerr << "Application - Start - Module " << (*it)->GetName() << " did not start correctly" << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool Application::Update()
{
	// Preupdate
	for (std::vector<Module*>::iterator it = modules.begin(); it != modules.end(); it++)
	{
		if ((*it)->Active())
		{
			if (!(*it)->PreUpdate())
			{
				std::cerr << "Application - Update - Module " << (*it)->GetName() << " failed preupdate" << std::endl;
				return false;
			}
		}
	}

	// Update
	for (std::vector<Module*>::iterator it = modules.begin(); it != modules.end(); it++)
	{
		if ((*it)->Active())
		{
			if (!(*it)->Update())
			{
				std::cerr << "Application - Update - Module " << (*it)->GetName() << " failed update" << std::endl;
				return false;
			}
		}
	}

	// Postupdate
	for (std::vector<Module*>::iterator it = modules.begin(); it != modules.end(); it++)
	{
		if ((*it)->Active())
		{
			if (!(*it)->PostUpdate())
			{
				std::cerr << "Application - Update - Module " << (*it)->GetName() << " failed postupdate" << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool Application::CleanUp()
{
	for (std::vector<Module*>::reverse_iterator it = modules.rbegin(); it != modules.rend(); it++)
	{
		if ((*it)->Active())
		{
			if (!(*it)->CleanUp())
			{
				std::cerr << "Application - CleanUp - Module " << (*it)->GetName() << " did not clean up correctly" << std::endl;
				return false;
			}

			(*it)->SetActive(false);
		}
	}
	return true;
}