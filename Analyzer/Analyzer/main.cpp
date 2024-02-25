#include "Core/Config.h"
#include <iostream>


int main() {
	Config config("../config.json", "0.0.0.0", 9002);
	config.show();
	std::string str = "xyzer";
	printf("test = %s", str);
}