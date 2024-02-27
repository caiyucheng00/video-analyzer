#ifndef __ALGORITHM__H_
#define __ALGORITHM__H_

#include <vector>
#include <string>

class Algorithm
{
public:
	virtual void imageClassify(int height, int width, unsigned char* bgr, std::vector<std::string>& classify) = 0;
};

#endif