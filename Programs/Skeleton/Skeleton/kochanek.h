#pragma once
#ifndef KOCHANEK_HEADER_INCLUDED
#define KOCHANEK_HEADER_INCLUDED

float derivative(const vec2& a, const vec2& b) {
	return (b.y - a.y) / (b.x - a.x);
}

class Kochanek : public GraphicsObject {
	std::vector<vec2>	controlPoints;
	const float dx = 0.01f;
	
	void refreshBuffer() {
		points.clear();
		addPoint(controlPoints[0]);
		const float firstX = controlPoints[0].x, lastX = controlPoints[controlPoints.size() - 1].x;
		for (float x = firstX; x < lastX; x += dx) {
			points.push_back(x);
			points.push_back(countCochanekY(x));
			addColor();
		}

		addPoint(vec2(10, -11));
		addPoint(vec2(-11, -11));

		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_DYNAMIC_DRAW);
	}

	

	float countCochanekY(const float x) {
		const unsigned int size = controlPoints.size();
		for (unsigned int i = 0; i < size; ++i) {
			if (controlPoints[i].x > x) {
				vec2 pi1 = controlPoints[i], pi = controlPoints[i - 1];
				vec2 v0, v1;
				if (i > 1) {
					vec2 pim1 = controlPoints[i - 2];
					v0 = (m(pim1, pi) + m(pi, pi1)) / 2;
				}
				else v0 = m(pi, pi1) / 2;
				if (i < size - 1) {
					vec2 pi2 = controlPoints[i + 1];
					v1 = (m(pi, pi1) + m(pi1, pi2)) / 2;
				}
				else v0 = m(pi, pi1) / 2;

				return hermite(pi, v0, pi.x, pi1, v1, pi1.x, x).y;
			}
		}
	}

	float m(const vec2& v1, const vec2& v2) {
		return (v2.y - v2.y) / (v2.x - v1.x);
	}

	vec2 hermite(const vec2& p0, const vec2& v0, float t0, const vec2& p1, const vec2& v1, float t1, float t) {
		vec2 a0 = p0, a1 = v0;
		float mT = t1 - t0;
		float mT2 = mT * mT;
		float mT3 = mT2 * mT;
		vec2 a2 = (p1 - p0) * (1 / mT2) * 3 - (v1 + v0 * 2)  * (1 / mT);
		vec2 a3 = (p0 - p1) * (1 / mT3) * 2 + (v1 + v0) * (1 / mT2);
		t -= t0;
		return a3 * pow(t, 3) + a2 * pow(t, 2) + a1 * t + a0;
	}

	float pow(const float num, const int exp) {
		float res = 1;
		for (int i = 0; i < exp; ++i) res *= num;
		return res;
	}

public:
	Kochanek() {
		this->color = vec3(1, 1, 0);
	}
	void create() {
		controlPoints.push_back(vec2(-10, -5));
		controlPoints.push_back(vec2(11, -5));

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
		refreshBuffer();
	}

	float getDerivative(const float x)const {
		const unsigned int size = points.size();
		for (unsigned int i = 0; i < size; i += 5) {
			if (points[i] > x) return (points[i + 1] - points[i - 4]) / dx;
		}
	}

	float getDistance(const float x, const float y) const {
		const unsigned int size = points.size();
		float distance = 0;
		for (unsigned int i = 0; i < size; i += 5) {
			if (points[i] > y ) return distance;
			if (points[i] > x) distance += length(vec2(points[i] - points[i - 5], points[i + 1] - points[i - 4]));
		}
	}

	vec3 getPosition(const float currentPositionX, const float dist, const bool dir = true) const {
		const unsigned int size = points.size();
		float distance = 0;
		if(dir)
			for (unsigned int i = 0; i < size; i += 5) {
				if (points[i] > currentPositionX) distance += length(vec2(points[i] - points[i - 5], points[i + 1] - points[i - 4]));
				if (distance > dist) return vec3(
					points[i],
					points[i + 1], 
					derivative(vec2(points[i - 5], points[i - 4]), vec2(points[i], points[i + 1])));
			}
		else 
			for (unsigned int i = size - 25; i >= 0; i -= 5) {
				if (points[i] < currentPositionX) distance += length(vec2(points[i + 5] - points[i], points[i + 6] - points[i + 1]));
				if (distance > dist) return vec3(
					points[i],
					points[i + 1],
					derivative(vec2(points[i + 5], points[i + 6]), vec2(points[i], points[i + 1])));
			}
	}

	void addControlPoint(float cX, float cY) {
		// input pipeline
		vec4 wVertex = vec4(cX, cY, 0, 1) * invM();
		// fill interleaved data
		const unsigned int size = controlPoints.size(); 
		std::vector<vec2>::iterator it;
		for (it = controlPoints.begin(); it != controlPoints.end(); ++it)
			if (wVertex.x < it->x)
				break;
		controlPoints.insert(it, vec2(wVertex.x, wVertex.y));
		for (vec2 c : controlPoints) printf("(%f %f) ", c.x, c.y);
		printf("\n");
		refreshBuffer();
	}

	void AddTranslation(const vec2& wT) {
		transformations.translate(wT + transformations.translation);
	}

	void draw() {
		GraphicsObject::draw();
		glDrawArrays(GL_LINE_LOOP, 0, points.size() / 5);
	}
};

#endif