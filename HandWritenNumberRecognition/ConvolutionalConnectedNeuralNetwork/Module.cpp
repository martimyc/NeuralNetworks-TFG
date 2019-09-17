#include "Module.h"

#include <iostream>

Module::Module(const std::string & name): name(name)
{}

Module::~Module()
{}

bool Module::Init()
{
	return true;
}

bool Module::Start()
{
	return true;
}

bool Module::PreUpdate()
{
	return true;
}

bool Module::Update()
{
	return true;
}

bool Module::PostUpdate()
{
	return true;
}

bool Module::CleanUp()
{
	return true;
}
