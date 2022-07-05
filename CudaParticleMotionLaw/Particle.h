#pragma once
#include <SFML/Graphics.hpp>

struct Particle
{
	sf::Vector2f position;
	int orientation;
	int neighbors;
};

