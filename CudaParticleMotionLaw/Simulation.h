#pragma once
#include "SFML/Graphics.hpp"

class Simulation
{
public:
	virtual void run() = 0;
private:
	virtual void pollEvent() = 0;
	virtual void update() = 0;
	virtual void draw() = 0;
protected:
	sf::RenderWindow window;
};