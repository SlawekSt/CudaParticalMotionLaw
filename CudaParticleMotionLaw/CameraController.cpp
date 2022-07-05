#include "CameraController.h"

CameraController::CameraController(float zoomLevel) : zoomLevel(zoomLevel), panning(false), panningAnchor(sf::Vector2i(0, 0))
{}

bool CameraController::handleWindowEvent(sf::RenderWindow& window, sf::Event e)
{
	switch (e.type)
	{
	case sf::Event::Resized:
	{
		sf::View view = window.getView();
		view.setSize(static_cast<float>(e.size.width), static_cast<float>(e.size.height));
		view.zoom(zoomLevel);
		window.setView(view);
		return true;
		break;
	}
	case sf::Event::MouseMoved:
	{
		if (panning)
		{
			sf::View view = window.getView();
			sf::Vector2f pos = sf::Vector2f(sf::Mouse::getPosition(window) - this->panningAnchor);
			view.move(-1.0f * pos * zoomLevel);
			window.setView(view);
			panningAnchor = sf::Mouse::getPosition(window);
			return true;
		}
		break;
	}
	case sf::Event::MouseButtonPressed:
	{
		if (e.mouseButton.button == sf::Mouse::Middle)
		{
			panning = true;
			panningAnchor = sf::Mouse::getPosition(window);
			return true;
		}
		break;
	}
	case sf::Event::MouseButtonReleased:
	{
		if (e.mouseButton.button == sf::Mouse::Middle)
		{
			panning = false;
			return true;
		}
		break;
	}
	case sf::Event::MouseWheelMoved:
	{
		sf::View view = window.getView();
		if (e.mouseWheel.delta < 0)
		{
			view.zoom(2.0f);
			zoomLevel *= 2;
		}
		else
		{
			view.zoom(0.5f);
			zoomLevel *= 0.5;
		}
		window.setView(view);
		return true;
	}
	}
	return false;
}

std::pair<sf::Vector2f, sf::Vector2f> CameraController::getViewBox(const sf::RenderWindow& window)
{
	sf::View view = window.getView();
	sf::Vector2f winStart = sf::Vector2f{ view.getCenter().x - view.getSize().x / 2,view.getCenter().y - view.getSize().y / 2 };
	sf::Vector2f winEnd = sf::Vector2f{ view.getCenter().x + view.getSize().x / 2, view.getCenter().y + view.getSize().y / 2 };

	return { winStart,winEnd };
}
