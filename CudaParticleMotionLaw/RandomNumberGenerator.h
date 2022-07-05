#pragma once
#include <random>

class RandomNumberGenerator
{
public:
    static float randFloat(float min, float max)
    {
        std::uniform_real_distribution<float> distr(min, max);
        return distr(rng);
    }
    static int randInt(int min, int max)
    {
        std::uniform_int_distribution<int>distr(min, max);
        return distr(rng);
    }
private:
    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng;
};