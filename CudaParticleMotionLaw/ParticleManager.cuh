#pragma once
#include "Particle.h"

class ParticleManager
{
public:
	ParticleManager();
	~ParticleManager();
	void update();
	void spawnParticles(sf::Vector2f position, int particleNumber = 1);
	void draw(sf::RenderTarget& target);
	void switchMode();
private:
	// Containers
	std::vector<Particle> particleVec;
	Particle* cudaParticle;
	std::vector<sf::Vector2f> positionVec;
	// Settings
	float particleSpeed;
	int alpha;
	int beta;
	float reactRadius;
	bool gpuBoost{ false };
};