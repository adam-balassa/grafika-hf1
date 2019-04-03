#ifndef WORLD_HEADER_INCLUDED
#define WORLD_HEADER_INCLUDED
class GraphicsObject;

GPUProgram gpuProgram;

class World {
public:
	Camera2D camera;		// 2D camera
	long tme = glutGet(GLUT_ELAPSED_TIME);
	std::vector<GraphicsObject*> objects;

	World() {
		// create program for the GPU
		gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	}

	GLuint programId() {
		return gpuProgram.getId();
	}

	float timeElapsed(long time) {
		long dT = time - tme;
		tme = time;
		return (float)dT / 100.0f;
	}

	void draw();

	void addObject(GraphicsObject* newObject);
};

#include "graphicsObject.h"

void World::draw() {
	for (GraphicsObject* object : objects)
		object->draw();
}

void World::addObject(GraphicsObject* newObject) {
	objects.push_back(newObject);
	newObject->setWorld(this);
	newObject->create();
}


World* world;

#endif



