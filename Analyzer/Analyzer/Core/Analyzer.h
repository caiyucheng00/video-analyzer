#ifndef __ANALYZER__H_
#define __ANALYZER__H_

#include <string>
#include <vector>

class Config;
class Control;
class Algorithm;

class Analyzer
{
public:
	explicit Analyzer(Config* config, Control* control);
	~Analyzer();

	void checkVideoFrame(bool check, unsigned char* data);

private:
	Config* _config;
	Control* _control;
	Algorithm* _algorithm;

	std::string classify_result;
	std::string show_str = "当前小麦生育期： ";
	std::string result_str;
};

#endif