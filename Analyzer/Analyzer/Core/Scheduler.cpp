#include "Scheduler.h"

Scheduler::Scheduler(Config* config) :
	_config(config),
	_state(false)
{

}

Scheduler::~Scheduler()
{

}

Config* Scheduler::getConfig()
{
	return _config;
}

void Scheduler::setState(bool state)
{

}

bool Scheduler::getState()
{

}

void Scheduler::loop()
{

}

int Scheduler::apiControls(std::vector<Control*>& controls)
{
	return 0;
}

Control* Scheduler::apiControl(std::string& code)
{
	return nullptr;
}

void Scheduler::apiControlAdd(Control* control, int& result_code, std::string& result_msg)
{

}

void Scheduler::apiControlCancel(Control* control, int& result_code, std::string& result_msg)
{

}
