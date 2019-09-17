#ifndef MODULE
#define MODULE

#include <string>

class Module
{
public:
	Module(const std::string& name);
	~Module();

	// Utility
	virtual bool Init();
	virtual bool Start();
	virtual bool PreUpdate();
	virtual bool Update();
	virtual bool PostUpdate();
	virtual bool CleanUp();

	// Getters & Setters
	inline bool Active() const { return active; }
	inline void SetActive(bool active) { this->active = active; }
	inline const std::string& GetName() const { return name; }

private:
	std::string name;
	bool active;
};

#endif //!MODULE