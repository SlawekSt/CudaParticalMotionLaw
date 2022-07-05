#include "ResourcesManager.h"
#include <errno.h>
#include "Error.h"

bool CheckLua(lua_State* L, int r)
{
	if (r != LUA_OK)
	{
		std::string errormsg = lua_tostring(L, -1);
		std::cout << errormsg << std::endl;
		return false;
	}
	return true;
}

ResourcesManager::ResourcesManager(std::string filename)
{
	L = luaL_newstate();
	luaL_openlibs(L);
	if (!CheckLua(L, luaL_dofile(L, filename.c_str())))
	{
		throw std::ios_base::failure("Configuration file: " + filename + " doesnt exists");
	}
}

ResourcesManager::~ResourcesManager()
{
	lua_close(L);
}

std::string ResourcesManager::getString(std::string name)
{
	lua_getglobal(L, name.c_str());
	if (lua_isstring(L, -1))
	{
		return lua_tostring(L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is string");
	}
}

int ResourcesManager::getInt(std::string name)
{
	lua_getglobal(L, name.c_str());
	if (lua_isnumber(L, -1))
	{
		return (int)lua_tointeger(L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is int");
	}
}

float ResourcesManager::getFloat(std::string name)
{
	lua_getglobal(L, name.c_str());
	if (lua_isnumber(L, -1))
	{
		return (float)lua_tonumber(L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is float");
	}
}