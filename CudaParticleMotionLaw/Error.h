#pragma once
#include <exception>
#include <string>

class InvalidTypeError : public std::exception
{
public:
	InvalidTypeError(const char* msg) : std::exception(msg) {}
};