#include "ParticleManager.cuh"
#include "ResourcesManager.h"
#include "RandomNumberGenerator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaKernels.cuh"
#include <stdio.h>
#define PI 3.14159265

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

float getDistance(sf::Vector2f fp, sf::Vector2f sp)
{
	return static_cast<float>(sqrt(pow(sp.x - fp.x, 2) + pow(sp.y - fp.y, 2)));
}


ParticleManager::ParticleManager()
{
	ResourcesManager manager("ParticleConfig.lua");

	particleSpeed = manager.getFloat("ParticleSpeed");
	alpha = manager.getInt("Alpha");
	beta = manager.getInt("Beta");
	reactRadius = manager.getFloat("ReactRadius");

}

ParticleManager::~ParticleManager()
{
	// Memory is already cleared when switched to cpu
	if (gpuBoost)
	{
		cudaFree(&cudaParticle);
	}
}

void ParticleManager::update()
{
	if (!gpuBoost)
	{
		for (int i = 0; i < particleVec.size(); ++i)
		{
			int Nt = 0;
			int Rt = 0;
			int Lt = 0;
			for (int j = 0; j < particleVec.size(); ++j)
			{
				// Skip identyical cells
				if (i == j)
					continue;
				if (getDistance(particleVec[j].position, particleVec[i].position) < reactRadius)
				{
					Nt++;
					float radianBetween = atan2(particleVec[j].position.y - particleVec[i].position.y, particleVec[j].position.x - particleVec[i].position.x);
					int finalDegrees = static_cast<int>(radianBetween * 180 / PI);
					int anglediff = (particleVec[i].orientation - finalDegrees + 180 + 360) % 360 - 180;
					if (anglediff > 90 || anglediff < -90)
					{
						Lt++;
					}
					else
					{
						Rt++;
					}
				}
			}
			particleVec[i].neighbors = Nt;
			int currentAngle = particleVec[i].orientation;
			sf::Vector2f currentPosition = particleVec[i].position;

			int turnDirection = alpha + sgn<int>(Rt - Lt) * beta * Nt;
			turnDirection %= 360;
			currentAngle += turnDirection;
			if (currentAngle > 360)
			{
				currentAngle %= 360;
			}
			else if (currentAngle < 0)
			{
				currentAngle += 360;
			}
			particleVec[i].orientation = currentAngle;

			currentPosition.x = static_cast<float>(currentPosition.x + particleSpeed * sin(-(currentAngle * PI / 180.0f)));
			currentPosition.y = static_cast<float>(currentPosition.y + particleSpeed * cos(-(currentAngle * PI / 180.0f)));

			positionVec[i] = currentPosition;

		}
		for (int i = 0; i < particleVec.size(); ++i)
		{
			particleVec[i].position = positionVec[i];
		}
	}
	else
	{
		cudaUpdateParticle(cudaParticle, particleVec.size(), particleSpeed, alpha, beta, reactRadius);
	}
}

void ParticleManager::spawnParticles(sf::Vector2f position, int particleNumber)
{
	// Can add only in cpu mode
	if (!gpuBoost)
	{
		// Add random offset to spawning position in order to avoid having all particles stacked in same location which would lead to bad particle behaviour
		for (int i = 0; i < particleNumber; ++i)
		{
			particleVec.push_back(Particle{ sf::Vector2f(position.x + RandomNumberGenerator::randFloat(-5.0f, 5.0f), position.y + RandomNumberGenerator::randFloat(-5.0f, 5.0f)), RandomNumberGenerator::randInt(0, 360), 0 });
			positionVec.push_back(sf::Vector2f(0.0f, 0.0f));
		}
	}
}

void ParticleManager::draw(sf::RenderTarget& target)
{
	if (gpuBoost)
	{
		cudaMemcpy(particleVec.data(), cudaParticle, sizeof(Particle) * particleVec.size(), cudaMemcpyDeviceToHost);
	}
	sf::CircleShape shape;
	shape.setRadius(1.0f);
	shape.setOrigin(shape.getRadius(), shape.getRadius());
	shape.setFillColor(sf::Color::Green);

	for (const auto& particle : particleVec)
	{
		shape.setFillColor(sf::Color::Green);
		int n = particle.neighbors;
		if (n > 35)
		{
			shape.setFillColor(sf::Color::Yellow);
		}
		else if (n > 15 && n <= 35)
		{
			shape.setFillColor(sf::Color::Blue);
		}
		shape.setPosition(particle.position);
		target.draw(shape);
	}
}

void ParticleManager::switchMode()
{
	// Gpu to cpu
	if (gpuBoost)
	{
		cudaMemcpy(particleVec.data(), cudaParticle, sizeof(Particle) * particleVec.size(), cudaMemcpyDeviceToHost);
		cudaFree(cudaParticle);
	}
	// Cpu to gpu
	else
	{
		cudaMalloc(&cudaParticle, sizeof(Particle) * particleVec.size());
		cudaMemcpy(cudaParticle, particleVec.data(), sizeof(Particle) * particleVec.size(),cudaMemcpyHostToDevice);
	}
	gpuBoost = !gpuBoost;
}
