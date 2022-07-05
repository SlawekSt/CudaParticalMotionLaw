#include "CudaKernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define PI 3.14159265

const int threadNumber = 32;

__device__ float getCudaDistance(sf::Vector2f fp, sf::Vector2f sp)
{
	return static_cast<float>(sqrt(pow(sp.x - fp.x, 2) + pow(sp.y - fp.y, 2)));
}

template <typename T> __device__ int sgnCuda(T val) {
	return (T(0) < val) - (val < T(0));
}

__global__ void cudaUpdate(Particle* particles,sf::Vector2f* positions,unsigned size, float particleSpeed, int alpha, int beta, float reactRadius)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < size)
	{
		int Nt = 0;
		int Rt = 0;
		int Lt = 0;
		for (int j = 0; j < size; ++j)
		{
			// Skip identyical cells
			if (tid == j)
				continue;
			if (getCudaDistance(particles[j].position, particles[tid].position) < reactRadius)
			{
				Nt++;
				float radianBetween = atan2(particles[j].position.y - particles[tid].position.y, particles[j].position.x - particles[tid].position.x);
				int finalDegrees = static_cast<int>(radianBetween * 180 / PI);
				int anglediff = (particles[tid].orientation - finalDegrees + 180 + 360) % 360 - 180;
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
		particles[tid].neighbors = Nt;
		int currentAngle = particles[tid].orientation;
		sf::Vector2f currentPosition = particles[tid].position;

		int turnDirection = alpha + sgnCuda<int>(Rt - Lt) * beta * Nt;
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
		particles[tid].orientation = currentAngle;

		currentPosition.x = static_cast<float>(currentPosition.x + particleSpeed * sin(-(currentAngle * PI / 180.0f)));
		currentPosition.y = static_cast<float>(currentPosition.y + particleSpeed * cos(-(currentAngle * PI / 180.0f)));

		positions[tid] = currentPosition;
	}
}

__global__ void cudaUpdatePos(Particle* particles, sf::Vector2f* positions, unsigned size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < size)
	{
		particles[tid].position = positions[tid];
	}
}


void cudaUpdateParticle(Particle* particles,unsigned size, float particleSpeed, int alpha, int beta, float reactRadius)
{
	sf::Vector2f* positions;
	cudaMalloc(&positions, sizeof(sf::Vector2f) * size);
	cudaUpdate << <(size + threadNumber - 1)/threadNumber, threadNumber >> > (particles,positions,size, particleSpeed, alpha, beta, reactRadius);
	cudaUpdatePos << < (size + threadNumber - 1) / threadNumber, threadNumber >> > (particles, positions, size);
	cudaFree(positions);
}
