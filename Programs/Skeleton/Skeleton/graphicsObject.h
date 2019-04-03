#ifndef GRAPHICS_OBJECT_HEADER_INCLUDED
#define GRAPHICS_OBJECT_HEADER_INCLUDED



class GraphicsObject {
protected:
	GLuint vao, vbo;
	std::vector<float> points;
	vec3 color;
	World* world;
	Transformation transformations;
	mat4 M() {
		return transformations.getTransformationMatrix() *  world->camera.P() * world->camera.V();
	}

	mat4 invM() {
		return transformations.getInversTransformationMatrix() * world->camera.Pinv() * world->camera.Vinv();
	}

	void addColor() {
		points.push_back(color.x);
		points.push_back(color.y);
		points.push_back(color.z);
	}

	void addPoint(const vec2& newPoint) {
		points.push_back(newPoint.x);
		points.push_back(newPoint.y);
		addColor();
	}

public:
	virtual void create() = 0;
	virtual void draw() {
		mat4 MVPTransform = M() ;
		MVPTransform.SetUniform(world->programId(), "MVP");
		glBindVertexArray(vao);
	};
	void setWorld(World* w) { this->world = w; }
	std::vector<float>& getPoints() { return points; }
	static float sqrt(const float x) {
		float top = x, bottom = 0, middle = x / 2;
		for (unsigned int i = 0; i < 20; ++i) {
			if (x > middle * middle)   bottom = middle;
			else top = middle;
			middle = (top + bottom) / 2;
		}
		return middle;
	}
};


#endif