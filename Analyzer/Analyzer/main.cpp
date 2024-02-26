#include "Core/Config.h"
#include "Core/Scheduler.h"
#include "Core/Server.h"


int main(int argc, char** argv) {
	Config config("../config.json", "0.0.0.0", 9002);
	config.show();

	Scheduler scheduler(&config);
	Server server;
	server.start(&scheduler);
	scheduler.loop();
}