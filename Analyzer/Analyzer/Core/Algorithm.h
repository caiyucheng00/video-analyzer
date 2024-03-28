#ifndef __ALGORITHM__H_
#define __ALGORITHM__H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
struct AlgorithmResult;

class Algorithm
{
public:
	Algorithm();
	virtual ~Algorithm();

	virtual void doAlgorithm(cv::Mat image, std::vector<AlgorithmResult>& results) = 0;
};

#endif