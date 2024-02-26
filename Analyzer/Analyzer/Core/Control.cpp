#include "Control.h"

bool Control::validateAdd(std::string& result_msg)
{
	if (code.empty() || streamUrl.empty() || behaviorCode.empty()) { // 缺失必须参数
		result_msg = "validate parameter error";
		return false;
	}

	if (pushStream) {
		if (pushStreamUrl.empty()) {  // 需要推流却缺失推流参数
			result_msg = "validate parameter pushStreamUrl is error: " + pushStreamUrl;
			return false;
		}
	}

	result_msg = "validate success";
	return true;
}

bool Control::validateCancel(std::string& result_msg)
{
	if (code.empty()) {  // 缺失必须参数
		result_msg = "validate parameter error";
		return false;
	}

	result_msg = "validate success";
	return true;
}
