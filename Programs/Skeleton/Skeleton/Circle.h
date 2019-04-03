#pragma once
#ifndef CIRCLE_HEADER_INCLUDED
#define CIRCLE_HEADER_INCLUDED

class Circle : public GraphicsObject {
protected:
	float r;
	bool filled = false;
	vec2 center;
	float scaleRatio = 0.1f;
	
	void generatePoints() {
		points.clear();
		float dx = 0.1f;
		for (float x = center.x - r; x < center.x + r; x += dx) {
			points.push_back(x);
			points.push_back(getY(x) + center.y);
			addColor();
		}

		for (float x = center.x + r; x > center.x - r; x -= dx) {
			points.push_back(x);
			points.push_back(-1 * getY(x) + center.y);
			addColor();
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_STATIC_DRAW);
	}

	float getY(float x) {
		x = x - center.x;
		return sqrt(r * r - x * x);
	}
public:
	Circle(float radius, vec3 color) {
		r = radius * 10;
		transformations.scale(scaleRatio);
		center = vec2(0, 0);
		this->color = color;
	}

	Circle(float radius, vec3 color, bool filled) : Circle(radius, color) {
		this->filled = filled;
	}

	void setCenter(const vec2& center) {
		this->center = center * 10;
		generatePoints();
	}

	void scale(const float scale) {
		this->scaleRatio *= scale;
		transformations.scale(this->scaleRatio);
	}

	void replace(const vec2& v) {
		transformations.translate(v);
	}

	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		generatePoints();
	}

	void draw() {
		GraphicsObject::draw();
		glDrawArrays(filled ? GL_TRIANGLE_FAN : GL_LINE_LOOP, 0, points.size() / 5);
	}
};

#endif#pragma once
