#pragma once
#include <string>
extern "C"
{
#include "Lua542/include/lua.h"
#include "Lua542/include/lauxlib.h"
#include "Lua542/include/lualib.h"
}
#include <iostream>
#ifdef _WIN32
#pragma comment(lib,"Lua542/liblua54.a")
#endif // _WIN32

class ResourcesManager
{
public:
	ResourcesManager(std::string filename);
	~ResourcesManager();

	std::string getString(std::string name);
	int getInt(std::string name);
	float getFloat(std::string name);
private:
	lua_State* L;
};
