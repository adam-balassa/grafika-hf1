#ifndef LINESTRIP_HEADER_INCLUDED
#define LINESTRIP_HEADER_INCLUDED

class LineStrip : public GraphicsObject {
	GLuint				vao, vbo;	// vertex array object, vertex buffer object
	std::vector<float>  vertexData; // interleaved data of coordinates and colors
public:
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
	}


	void AddPoint(float cX, float cY) {
		// input pipeline
		vec4 wVertex = vec4(cX, cY, 0, 1) * invM();
		// fill interleaved data
		vertexData.push_back(wVertex.x);
		vertexData.push_back(wVertex.y);
		vertexData.push_back(1); // red
		vertexData.push_back(1); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void AddTranslation(vec2 wT) { 
		transformations.translate(wT + transformations.translation);
	}

	void draw() {
		if (vertexData.size() > 0) {
			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			mat4 MVPTransform = M()/* world->camera.V() * world->camera.P()*/;
			MVPTransform.SetUniform(world->programId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
		}
	}
};

#endif