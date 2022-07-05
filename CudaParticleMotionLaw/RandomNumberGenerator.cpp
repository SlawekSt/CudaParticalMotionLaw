#include "RandomNumberGenerator.h"

thread_local std::random_device RandomNumberGenerator::rd;
thread_local std::mt19937 RandomNumberGenerator::rng(rd());