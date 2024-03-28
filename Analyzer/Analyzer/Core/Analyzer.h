#ifndef __ANALYZER__H_
#define __ANALYZER__H_

#include <string>
#include <vector>

class Config;
class Control;
class Algorithm;
class Model;

struct AlgorithmResult
{
	std::string strategy_name;
	std::string class_name;
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
};

class Analyzer
{
public:
	explicit Analyzer(Config* config, Control* control, Model* model);
	~Analyzer();

	void checkVideoFrame(bool check, unsigned char* data);

private:
	Config* _config;
	Control* _control;
	Algorithm* _algorithm;
	std::vector<AlgorithmResult> _results;

	std::string show_str = "当前小麦生育期： ";
	std::string result_str;
};

#endif