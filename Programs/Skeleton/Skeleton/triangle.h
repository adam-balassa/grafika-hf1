#ifndef TRIANGLE_HEADER_INCLUDED
#define TRIANGLE_HEADER_INCLUDED
#include "graphicsObject.h"
class Triangle : public GraphicsObject {
	struct Point2D; struct Point2D {
		float x, y;
		float r, g, b, a;
		Point2D(float x, float y, float r, float g, float b, float a = 1)
			:x(x), y(y), r(r), g(g), b(b), a(a) {}

		void copyXY(float* array, unsigned int pos) {
			array[pos] = x;
			array[pos + 1] = y;
		}

		void copyRGB(float* array, unsigned int pos) {
			array[pos] = r;
			array[pos + 1] = g;
			array[pos + 2] = b;
		}
	}; 
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	vec2 wTranslate;	// translation
	float phi;			// angle of rotation
	std::vector<Triangle::Point2D> points;
public:
	Triangle() { 
		points.push_back(Point2D(-8, 8, 1, 0, 0));
		points.push_back(Point2D(-6, -10, 0, 1, 0));
		points.push_back(Point2D(8, -2, 0, 0, 1));

		Animate(0); 
	}

	void create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[6];
		for (int i = 0; i < 3; ++i) points[i].copyXY(vertexCoords, i * 2);

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		float vertexColors[9];
		for (int i = 0; i < 3; ++i) points[i].copyRGB(vertexColors, i * 3);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU
		glEnableVertexAttribArray(1);  
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		transformations.rotate(t);
	}

	void draw() {
		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		M().SetUniform(world->programId(), "MVP");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
	
};
#endif