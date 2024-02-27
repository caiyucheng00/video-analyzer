#ifndef __ANALYZER__H_
#define __ANALYZER__H_

class Scheduler;
class Control;

class Analyzer
{
public:
	explicit Analyzer(Scheduler* scheduler, Control* control);
	~Analyzer();
};

#endif