#pragma once
#include <SFML/Graphics.hpp>

class CameraController
{
public:
	CameraController(float zoomLevel = 1.0f);
	bool handleWindowEvent(sf::RenderWindow& window, sf::Event e);
	std::pair<sf::Vector2f, sf::Vector2f> getViewBox(const sf::RenderWindow& window);
private:
	float zoomLevel;
	bool panning;
	sf::Vector2i panningAnchor;
};