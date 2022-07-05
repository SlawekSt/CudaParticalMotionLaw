#pragma once

#include "Simulation.h"
#include "CameraController.h"
#include "ParticleManager.cuh"

class ParticleSimulation : public Simulation
{
public:
	ParticleSimulation();
	void run() override;
private:
	void pollEvent() override;
	void update() override;
	void draw() override;
private:
	bool pause{ true };
	CameraController camera;
	ParticleManager particleManager;
};