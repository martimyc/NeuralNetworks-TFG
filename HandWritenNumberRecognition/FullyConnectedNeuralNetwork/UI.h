#ifndef UI_
#define UI_

#include "Module.h"

class ImGuiContext;

class UI : public Module
{
public:
	UI();
	~UI();

	// Functionality
	bool Init() override;
	bool Start() override;
	bool PreUpdate() override;
	bool Update() override;
	bool PostUpdate() override;
	bool CleanUp() override;


private:

};

#endif // !UI_

