#ifndef __ANALYZER__H_
#define __ANALYZER__H_

#include <string>
#include <vector>

class Scheduler;
class Control;

class Analyzer
{
public:
	explicit Analyzer(Scheduler* scheduler, Control* control);
	~Analyzer();

	void checkVideoFrame(bool check, int64_t frameCount, unsigned char* data);
	void checkAudioFrame(bool check, int64_t frameCount, unsigned char* data, int size);

};

#endif