#pragma once
#include "Particle.h"


void cudaUpdateParticle(Particle* particles, unsigned size, float particleSpeed, int alpha, int beta, float reactRadius);
