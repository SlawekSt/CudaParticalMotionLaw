#include "ParticleSimulation.h"
#include "ResourcesManager.h"
#include <chrono>


ParticleSimulation::ParticleSimulation()
{
	ResourcesManager manager("WindowConfig.lua");

	window.create(sf::VideoMode(manager.getInt("WindowWidth"), manager.getInt("WindowHeight")), manager.getString("WindowTitle"));
	window.setFramerateLimit(manager.getInt("Framerate"));
}

void ParticleSimulation::run()
{
	while (window.isOpen())
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		update();
		window.clear(sf::Color::White);
		draw();
		window.display();
		pollEvent();
		auto t_end = std::chrono::high_resolution_clock::now();
		double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
		std::cout << "Frame: " << elapsed_time_ms << std::endl;
	}
}

void ParticleSimulation::pollEvent()
{
	sf::Event e;
	while (window.pollEvent(e))
	{
		if (camera.handleWindowEvent(window, e))
		{
			continue;
		}
		if (e.type == sf::Event::Closed)
		{
			window.close();
			break;
		}
		if (e.type == sf::Event::KeyPressed)
		{
			if (e.key.code == sf::Keyboard::P)
			{
				pause = !pause;
			}
			if (e.key.code == sf::Keyboard::R)
			{
				particleManager.switchMode();
			}
		}
		if (e.type == sf::Event::MouseButtonPressed)
		{
			if (e.mouseButton.button == sf::Mouse::Left)
			{
				particleManager.spawnParticles(window.mapPixelToCoords(sf::Mouse::getPosition(window)),100);
			}
		}
	}
}

void ParticleSimulation::update()
{
	particleManager.update();
}

void ParticleSimulation::draw()
{
	particleManager.draw(window);
}
