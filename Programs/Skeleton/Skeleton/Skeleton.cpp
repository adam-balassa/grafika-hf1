//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Balassa Ádám
// Neptun : DXXEXO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float maxWidth = 10.0f;
const float minWidth = -10.0f;


const float PI = (float)M_PI;

inline vec2  m(const vec2& v1, const vec2& v2) {
	return (v2 - v1) * (1  / (v2.x - v1.x));
}
inline float  m2(const vec2& v1, const vec2& v2) {
	return (v2.y - v1.y) * (1 / (v2.x - v1.x));
}

class Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
		layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	
		out vec3 color;									// output attribute

		void main() {
			color = vertexColor;														// copy color from input to output
			gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in vec3 color;				// variable input: interpolated color of vertex shader
		out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

		void main() {
			fragmentColor = vec4(color, 1); // extend RGB to RGBA
		}
	)";

	const char * textureVertexShader = R"(
		#version 330
		precision highp float;

		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		out vec2 texCoord;								// output attribute

		void main() {
			texCoord = (vertexPosition + vec2(1, 1)) / 2;						// from clipping to texture space
			gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// already in clipping space
		}
	)";

	const char * textureFragmentShader = R"(
		#version 330
		precision highp float;

		uniform sampler2D textureUnit;

		in vec2 texCoord;			// variable input: interpolated texture coordinates
		out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
		
		void main() {
			fragmentColor = texture(textureUnit, texCoord);
		}
	)";


	GPUProgram simpleProgram;
	GPUProgram textureProgram;
public: 
	void initializeShaders() {
		simpleProgram.Create(vertexSource, fragmentSource, "fragmentColor"); 
		textureProgram.Create(textureVertexShader, textureFragmentShader, "fragmentColor");
		setShader(false);
	}

	void setShader(bool texture) {
		if (texture) textureProgram.Use();
		else simpleProgram.Use();
	}

	void setMVP(mat4 mvp) {
		char * str = "MVP";
		mvp.SetUniform(simpleProgram.getId(), str);
	}

	void bindTexture(const unsigned int textureId) {
		int location = glGetUniformLocation(textureProgram.getId(), "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
	}
};
class Transformation {
	bool rotated = false, translated = false, scaled = false, turned = true;

	mat4 unitM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 scaleM() const {
		return mat4(
			scaleX * (turned ? 1 : -1), 0, 0, 0,
			0, scaleY, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);
	}

	mat4 invscaleM() const {
		return mat4(
			1 / scaleX * (turned ? -1 : 1), 0, 0, 0,
			0, 1 / scaleY, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);
	}

	mat4 translateM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			translation.x, translation.y, 0, 1);
	}

	mat4 invtranslateM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			-translation.x, -translation.y, 0, 1);
	}

	mat4 rotateM() const {
		return mat4(
			cosf(rotation), sinf(rotation), 0, 0,
			-sinf(rotation), cosf(rotation), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 invrotateM() const {
		return mat4(cosf(rotation), -sinf(rotation), 0, 0,
			sinf(rotation), cosf(rotation), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

public:
	vec2 translation = vec2(0, 0);
	float rotation = 0;
	float scaleX = 1, scaleY = 1;

	mat4 getTransformationMatrix() const {
		mat4 M = unitM();
		if (rotated)
			M = M * rotateM();
		if (scaled)
			M = M * scaleM();
		if (translated)
			M = M * translateM();
		return M;
	}

	mat4 getInversTransformationMatrix() const {
		mat4 M = unitM();
		if (scaled)
			M = M * invscaleM();
		if (rotated)
			M = M * invrotateM();
		if (translated)
			M = M * invtranslateM();
		return M;
	}

	void rotate(float phi) {
		rotated = true;
		rotation = phi;
	}

	void translate(const vec2& tranlateVector) {
		translated = true;
		translation = tranlateVector;
	}

	void scale(float rate) {
		scale(rate, rate);
	}

	void scale(float rateX, float rateY) {
		scaled = true;
		scaleX = rateX;
		scaleY = rateY;
	}

	void turn() {
		turned = !turned;
		scale(scaleX, scaleY);
	}
};
class Mesh {
	GLenum staticDraw;
	GLenum drawingMethod;
	GLuint vao, colorVBO, positionVBO;
	unsigned int pointSize;
public:
	Mesh(GLenum drawingMethod = GL_LINE_STRIP, GLenum staticDraw = GL_STATIC_DRAW): 
		drawingMethod(drawingMethod), staticDraw(staticDraw) {
		initialize();
	}

	void initialize() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &positionVBO); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr); // attribute array, components/attribute, component type, normalize?, stride, offset
		
																										
		glGenBuffers(1, &colorVBO); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, colorVBO); // Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
		glBindVertexArray(0);
	}


	void setPositions(const std::vector<float>& positions) {
		glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
		glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), &positions[0], staticDraw);
		pointSize = positions.size() / 2;
	}

	void setPositions(const std::vector<vec2>& positions) {
		std::vector<float> points;
		for (const vec2 vec : positions) {
			points.push_back(vec.x);
			points.push_back(vec.y);
		}
		setPositions(points);
	}

	void setColors(const std::vector<float>& colors) {
		glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
		glBufferData(GL_ARRAY_BUFFER, pointSize * 3 * sizeof(float), &colors[0], staticDraw);
	}

	void setColors(const std::vector<vec3>& colors) {
		std::vector<float> realColors;
		for (const vec3 vec : colors) {
			realColors.push_back(vec.x);
			realColors.push_back(vec.y);
			realColors.push_back(vec.z);
		}
		setColors(realColors);
	}

	void setUniformColor(const vec3& color) {
		std::vector<float> colors;
		for (unsigned int i = 0; i < pointSize; ++i) {
			colors.push_back(color.x);
			colors.push_back(color.y);
			colors.push_back(color.z);
		}
		setColors(colors);
	}

	void draw() {
		glBindVertexArray(vao); 
		glDrawArrays(drawingMethod, 0, pointSize);
	}
};
class TextureMesh {
	unsigned int vao, vbo, textureId;	// vertex array object id and texture id
	vec2 vertices[4];
public:
	TextureMesh() {
		vertices[0] = vec2(-1, -1);
		vertices[1] = vec2(1, -1);
		vertices[2] = vec2(1, 1);
		vertices[3] = vec2(-1, 1);
	}
	void initialize() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		// Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
	}

	void setImage(const std::vector<vec3>& image, const unsigned int width, const unsigned int height){
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	unsigned int getTextureId() {
		return textureId;
	}

	void draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
class CatmullRom {
	std::vector<vec2> controlPoints;
	const float dx;
	const float tension;

	float catmull(const float x) {
		const unsigned int size = controlPoints.size();
		for (unsigned int i = 1; i < size; ++i) {
			if (controlPoints[i].x > x) {
				vec2 pi1 = controlPoints[i], pi = controlPoints[i - 1];
				vec2 v0, v1;
				if (i > 1) {
					vec2 pim1 = controlPoints[i - 2];
					v0 = (m(pim1, pi) + m(pi, pi1));
				}
				else v0 = m(pi, pi1);
				if (i < size - 1) {
					vec2 pi2 = controlPoints[i + 1];
					v1 = (m(pi, pi1) + m(pi1, pi2));
				}
				else v1 = m(pi, pi1);
				v0 = v0 * 0.5f * (1 - tension);
				v1 = v1 * 0.5f * (1 - tension);
				return hermite(pi, v0, pi.x, pi1, v1, pi1.x, x).y;
			}
		}
	}

	inline vec2 hermite(const vec2& p0, const vec2& v0, float t0, const vec2& p1, const vec2& v1, float t1, float t) {
		vec2 a0 = p0, a1 = v0;
		float mT = t1 - t0;
		float mT2 = mT * mT;
		float mT3 = mT2 * mT;
		vec2 a2 = (p1 - p0) * (1 / mT2) * 3 - (v1 + v0 * 2)  * (1 / mT);
		vec2 a3 = (p0 - p1) * (1 / mT3) * 2 + (v1 + v0) * (1 / mT2);
		t -= t0;
		return a3 * pow(t, 3) + a2 * pow(t, 2) + a1 * t + a0;
	}

	inline float pow(const float num, const int exp) {
		float res = 1;
		for (int i = 0; i < exp; ++i) res *= num;
		return res;
	}
public:
	CatmullRom(const std::vector<vec2>& controlPoints, const float tension, const float dx) :
		tension(tension),
		dx(dx) {
		std::vector<vec2> cps;
		cps.push_back(vec2(controlPoints[0].x - 1.0f, controlPoints[0].y));
		for (vec2 p : controlPoints) cps.push_back(p);
		cps.push_back(vec2(controlPoints[controlPoints.size() - 1].x + 1.0f, controlPoints[controlPoints.size() - 1].y));
		this->controlPoints = cps;
	}

	void setSpline(std::vector<vec2>& topPoints) {
		topPoints.clear();
		const float firstX = controlPoints[0].x - 4.0f, lastX = controlPoints[controlPoints.size() - 1].x + 4.0f;
		for (float x = firstX; x < lastX; x += dx)
			topPoints.push_back(vec2(x, catmull(x)));
	}
};
class BackGround {
	TextureMesh* texture;
	const unsigned int width = 1200, height = 1200;
public:

	float isUnderSpline(const std::vector<vec2>& points, const vec2& pos) {
		for (const vec2 point : points) {
			if (point.x > pos.x)
				return pos.y - point.y;
		}
		return true;
	}

	void init() {
		texture = new TextureMesh();
		texture->initialize();

		std::vector<vec2> cps;
		cps.push_back(vec2(-15.0f, 0.0f));
		cps.push_back(vec2(-5.0f, 7.0f));
		cps.push_back(vec2(0.0f, 0.0f));
		cps.push_back(vec2(5.0f, 5.0f));
		cps.push_back(vec2(15.0f, 0.0f));
		CatmullRom* spline = new CatmullRom(cps, -1, 0.05f);

		std::vector<vec2> points;
		spline->setSpline(points);
		std::vector<vec3> image(width * height);
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				const vec2 point = vec2((float)x - width / 2.0f, (float)y - height / 2.0f) * (1.0f / height) * 20.0f;
				float d = isUnderSpline(points, point);
				if (d > 0) image[y * width + x] = vec3(0.2f, 0.4f, 0.9f) + vec3(0.1f, 0.1f, 0.1f) * (3.5f - d / 3);
				else {
					d += point.y / 2.0f;
					if (d > -3.0) {
						image[y * width + x] = vec3(0.25f, 0.45f, 0.15f);
						continue;
					}
					d += point.y / 1.5f;
					if (d > -9.0f) image[y * width + x] = vec3(0.2f, 0.35f, 0.1f);
					else image[y * width + x] = vec3(0.1f, 0.25f, 0.05f);
				}
			}
		}
		delete spline;
		texture->setImage(image, width, height);
	}

	unsigned int getTexture() {
		return texture->getTextureId();
	}

	void draw() {
		texture->draw();
	}

	virtual ~BackGround() {
		delete texture;
	}
};
class GraphicsObject {
protected:
	GraphicsObject* container = nullptr;
	Transformation transformations;
	Mesh* mesh = nullptr;
public:
	GraphicsObject() { }

	virtual void init() = 0;

	void rotate(const float phi) {
		transformations.rotate(phi);
	}

	void translate(const vec2& tranlateVector) {
		transformations.translate(tranlateVector);
	}

	void scale(float rate) {
		transformations.scale(rate);
	}

	void turn() {
		transformations.turn();
	}

	mat4 M() const {
		mat4 matrix = transformations.getTransformationMatrix();
		if (container != nullptr)
			return matrix * container->M();
		return matrix;
	}

	mat4 invM() const {
		mat4 matrix = transformations.getInversTransformationMatrix();
		if (container != nullptr)
			return matrix * container->invM();
		return matrix;
	}

	void setContainer(GraphicsObject* parent) {
		container = parent;
	}

	virtual void draw() {
		if(mesh != nullptr)
			mesh->draw();
	}
};
class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(20, 20) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = t; }
};
class Renderer {
	std::vector<GraphicsObject*> objects;
	BackGround bg;
	Shader* shader;
public:
	Camera2D camera;		// 2D camera

	Renderer(Shader* shader): shader(shader) { 
		shader->initializeShaders();
		bg.init();
	}

	void render() {
		shader->setShader(true);
		shader->bindTexture(bg.getTexture());
		bg.draw();

		shader->setShader(false);
		for (GraphicsObject* object : objects) {
			shader->setMVP(object->M() * camera.P() * camera.V());
			object->draw();
		}
	}

	void addObject(GraphicsObject* obj) {
		objects.push_back(obj);
		obj->init();
	}

	virtual ~Renderer() {
		delete shader;
	}
};
Renderer* renderer;
class Circle : public GraphicsObject {
	const float radius; 
	const vec3 color;
	const bool filled;
public:
	Circle(const float radius, const vec3& color = vec3(1, 1, 1), const bool filled = false)
		: radius(radius), color(color), filled(filled) { 
		
	}

	void init() {
		mesh = new Mesh(filled ? GL_TRIANGLE_FAN : GL_LINE_LOOP, GL_STATIC_DRAW);
		std::vector<float> points;
		if (filled) {
			points.push_back(0);
			points.push_back(0);
		}
		const float dPhi = 0.15f;
		for (float phi = 0; phi <= 2 * PI + dPhi; phi += dPhi) {
			points.push_back(sinf(phi) * radius);
			points.push_back(cosf(phi) * radius);
		}
		mesh->setPositions(points);
		mesh->setUniformColor(color);
	}

};
class WideCircle : public GraphicsObject{
	const float radiusOut, width;
	const vec3 colorIn, colorOut;
public:
	WideCircle(const float radiusOut, const float width, const vec3& colorIn, const vec3& colorOut) 
		: radiusOut(radiusOut), width(width), colorIn(colorIn), colorOut(colorOut) { }

	void init() {
		mesh = new Mesh(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
		std::vector<float> points; 
		std::vector<vec3> colors;
		const float radiusIn = radiusOut - width;
		const float dPhi = 0.15f;
		for (float phi = 0; phi <= 2 * PI + dPhi; phi += dPhi) {
			points.push_back(sinf(phi) * radiusOut);
			points.push_back(cosf(phi) * radiusOut);
			points.push_back(sinf(phi) * radiusIn);
			points.push_back(cosf(phi) * radiusIn);
			colors.push_back(colorIn);
			colors.push_back(colorOut);
		}
		mesh->setPositions(points);
		mesh->setColors(colors);
	}
};
class Rect : public GraphicsObject {
	const float a, b;
	const vec3 colorIn, colorOut;
public:
	Rect(const float a, const float b, const vec3& colorIn, const vec3& colorOut)
		: a(a), b(b), colorIn(colorIn), colorOut(colorOut) { }

	void init() {
		mesh = new Mesh(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
		std::vector<vec2> points;
		std::vector<vec3> colors;
		const float dY = b / 2.0f;
		for (float y = 0; y <= b; y += dY) {
			points.push_back(vec2(0, -y));
			points.push_back(vec2(a, -y));

			colors.push_back(colorIn);
			colors.push_back(colorOut);
		}
		mesh->setPositions(points);
		mesh->setColors(colors);
	}
};

class Mountain : public GraphicsObject {
protected:
	std::vector<vec2> controlPoints;
	std::vector<vec2> topPoints;

	virtual void refresh() {
		vec3 topColor(0.3f, 0.7f, 0.25f);
		vec3 bottomColor(0.08f, 0.13f, 0.05f);
		countTopPoints();
		std::vector<float> points;
		std::vector<vec3> colors;
		for (unsigned int i = 0; i < topPoints.size(); i += 100) {
			const vec2 point = topPoints[i];
			points.push_back(point.x); points.push_back(point.y);
			colors.push_back(topColor);

			points.push_back(point.x); points.push_back(-11);
			colors.push_back(bottomColor);
		}
		mesh->setPositions(points);
		mesh->setColors(colors);
	}

	void countTopPoints() {
		CatmullRom* spline = new CatmullRom(controlPoints, -1, 0.00003f);
		spline->setSpline(topPoints);
		delete spline;
	}
public:
	virtual void init() {
		mesh = new Mesh(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
		controlPoints.push_back(vec2(-13, -6));
		controlPoints.push_back(vec2(13, -6));
		refresh();
	}

	void addControlPoints(const float x, const float y) {
		Camera2D& camera = renderer->camera;
		vec4 wVertex = vec4(x, y, 0, 1)  * camera.Vinv() * camera.Pinv() * invM();

		const unsigned int size = controlPoints.size();
		std::vector<vec2>::iterator it;
		for (it = controlPoints.begin(); it != controlPoints.end(); ++it)
			if (wVertex.x < it->x) break;

		controlPoints.insert(it, vec2(wVertex.x, wVertex.y));
		refresh();
	}

	float getDerivative(const float x, bool direction = true) {
		const unsigned int size = topPoints.size();
		const unsigned int i = findPoint(x);
		return m2(topPoints[i], topPoints[i - 1]) * (direction ? 1.0f : -1.0f);
	}

	unsigned int findPoint(const float x) {
		const unsigned int size = topPoints.size();
		unsigned int top = size;
		unsigned int bottom = 0;
		while (true) {
			unsigned int middle = (top + bottom) / 2;
			if (topPoints[middle].x <= x && topPoints[middle - 1].x > x || middle == bottom)
				return middle;
			if (topPoints[middle].x < x) bottom = middle;
			else top = middle;
		}

	}

	vec2 getNextPosition(const vec2& currentPosition, float distance, bool direction = true) {
		const unsigned int p = findPoint(currentPosition.x);
		if (direction) {
			const unsigned int size = topPoints.size();
			float dist = 0;
			for (unsigned int i = p; i < size; ++i) {
				dist += length(topPoints[i] - topPoints[i - 1]);
				if (dist > distance)
					return topPoints[i - 1];
			}
		}
		else {
			const unsigned int size = topPoints.size();
			float dist = 0;
			for (int i = p - 1; i >= 0; --i) {
				dist += length(topPoints[i + 1] - topPoints[i]);
				if (dist > distance)
					return topPoints[i + 1];
			}
		}
		return currentPosition;
	}
};

Mountain ground;
class Biker : public GraphicsObject {
	enum Direction { LEFT = -1, RIGHT = 1 };
	float phi;
	const float radius = 1.7f;

	float velocity = 5.0f;
	Direction direction = RIGHT;
	vec2 position = vec2(0, -6.0f);
	
	class Wheel : public GraphicsObject {
		const float radius;
		const unsigned int spokes = 13;
		WideCircle* tire;
	public:
		Wheel(const float radius) : radius(radius) {}

		void init() {
			tire = new WideCircle(radius, 0.5f, vec3(0.35f, 0.18f, 0.1f), vec3(0.2f, 0.2f, 0.2f));
			tire->setContainer(this);
			renderer->addObject(tire);
			mesh = new Mesh(GL_LINES, GL_DYNAMIC_DRAW);
			std::vector<vec2> points;
			const float dPhi = PI / spokes * 2;
			const float r = radius - 0.5f;
			for (unsigned int i = 0; i < spokes; i++) {
				points.push_back(vec2(0, 0));
				points.push_back(vec2(sinf(i * dPhi) * r, cosf(i * dPhi) * r));
			}
			mesh->setPositions(points);
			mesh->setUniformColor(vec3(0.2f, 0.2f, 0.2f));
		}
		virtual ~Wheel() {
			delete tire;
		}

	};
	class StaticBody : public GraphicsObject {
		Circle* head;
	public:
		StaticBody() {}

		void init() {
			vec3 color = vec3(0.7f, 0.4f, 0.3f);
			head = new Circle(0.7f, color, true);
			head->setContainer(this);
			renderer->addObject(head);
			head->translate(vec2(0, 3.0f + 0.7f));

			mesh = new Mesh(GL_LINES, GL_DYNAMIC_DRAW);
			std::vector<vec2> points;
			points.push_back(vec2(0, 0)); points.push_back(vec2(0, 3.0f));
			points.push_back(vec2(0, 2.6f)); points.push_back(vec2(-0.4f, 1.0f));
			points.push_back(vec2(-0.4f, 1.0f)); points.push_back(vec2(0.8f, -0.3f));

			mesh->setPositions(points);
			mesh->setUniformColor(color);
		}
		virtual ~StaticBody() {
			delete head;
		}

	};
	class StaticSeat : public GraphicsObject {
		const vec2 bottomPos;
		Rect *column, *seat1, *seat2;
	public:
		StaticSeat(const vec2& bottomPos): bottomPos(bottomPos) {}

		void init() {
			const float width = 0.2f;
			column = new Rect(width, length(bottomPos), vec3(0.2f, 0.2f, 0.2f), vec3(0.4f, 0.4f, 0.4f));
			column->setContainer(this);
			renderer->addObject(column);
			column->translate(bottomPos - vec2(width/2, width));

			const float seatWidth = 1.4f;
			seat1 = new Rect(seatWidth, width, vec3(0.9f, 0.2f, 0.2f), vec3(0.4f, 0.1f, 0.1f));
			seat1->setContainer(this);
			renderer->addObject(seat1);
			seat1->translate(bottomPos - vec2(seatWidth / 2, width));

			seat2 = new Rect(width, seatWidth / 2, vec3(0.9f, 0.2f, 0.2f), vec3(0.4f, 0.1f, 0.1f));
			seat2->setContainer(this);
			renderer->addObject(seat2);
			seat2->translate(bottomPos - vec2(1.2f, -0.1f));
			seat2->rotate(PI / 4);
		}

		virtual ~StaticSeat() {
			delete column;
			delete seat1;
		}
	};
	class Leg : public GraphicsObject {
		const float thigh = 2.3f;
		const float pedalRadius = 0.6f;
		const vec2 bottomPos;

		vec2 knee(const vec2& footPos, const vec2& bottomPos) const {
			const vec2 m = vec2((footPos.x + bottomPos.x) / 2, (footPos.y + bottomPos.y) / 2);
			const vec2 norm = normalize(vec2(-footPos.y + bottomPos.y, footPos.x - bottomPos.x));
			const float c = length(footPos - bottomPos);
			const float d = sqrt(thigh * thigh - c * c / 4);
			return m + norm * d;
		}

		vec2 foot(const float phi) const {
			const vec2 pedalPos = vec2(0, 0);
			return vec2(pedalRadius * sin(phi) + pedalPos.x, pedalRadius * cos(phi) + pedalPos.y);
		}

		void countPoints(const float phi) {
			const vec2 footPos = foot(phi);
			const vec2 kneePos = knee(footPos, bottomPos);

			std::vector<vec2> points;
			points.push_back(bottomPos);
			points.push_back(kneePos);
			points.push_back(footPos);

			mesh->setPositions(points);
			mesh->setUniformColor(vec3(0.7f, 0.4f, 0.3f));
		}
	public:
		Leg(const vec2& bottomPos) : bottomPos(bottomPos) {}

		void init() {
			mesh = new Mesh(GL_LINE_STRIP, GL_DYNAMIC_DRAW);
			countPoints(0);
		}

		void ride(const float phi) {
			countPoints(phi);
		}
	};
	Wheel* wheel;
	StaticBody* body;
	StaticSeat* seat;
	Leg *right, *left;

	vec2 countCenter() {
		const float derivative = ground.getDerivative(position.x);
		const vec2 norm = normalize(vec2(-derivative, 1));
		return position + norm * radius * transformations.scaleX;
	}

public:
	void init() {
		const vec2 bottom = vec2(-0.5f, 3.5f);
		left = new Leg(bottom);
		left->setContainer(this);
		renderer->addObject(left);

		wheel = new Wheel(radius);
		wheel->setContainer(this);
		renderer->addObject(wheel);

		seat = new StaticSeat(vec2(0, bottom.y));
		seat->setContainer(this);
		renderer->addObject(seat);

		right = new Leg(bottom);
		right->setContainer(this);
		renderer->addObject(right);

		body = new StaticBody();
		body->setContainer(this);
		body->translate(bottom);
		renderer->addObject(body);

		scale(0.4f);
	}

	float a() {
		float a = 6.9f;
		const float derivative = ground.getDerivative(position.x, direction == LEFT ^ (velocity < 0));
		const float alpha = atanf(derivative);
		const float gravityForce = 9.0f, rho = 0.13f;
		a += gravityForce * sin(alpha);
		a -= (velocity * velocity * rho) * (velocity > 0 ? 1 : -1);
		return a;
	}

	void move(const float dT) {
		if (position.x < minWidth && direction == LEFT || position.x > maxWidth && direction == RIGHT)
			turn();
		velocity += a() * dT;
		const float distance = velocity * dT;
		phi += distance / radius;
		wheel->rotate(-phi);
		left->ride(phi);
		right->ride(phi + PI);

		position = ground.getNextPosition(position, abs(distance), (direction == RIGHT) ^ (distance < 0));
		translate(countCenter());
	}

	void reset() {
		velocity = 5.0f;
		position = vec2(0, 0);
		move(0.0001f);
	}

	void turn() {
		GraphicsObject::turn();
		if (direction == LEFT)	direction = RIGHT;
		else					direction = LEFT;
	}

	vec2 getPosition() {
		return transformations.translation;
	}

	virtual ~Biker() {
		delete wheel;
		delete body;
		delete right; delete left;
	}
};


Biker biker;
bool follow = false;

void toggleFollowing() {
	follow = !follow;
	renderer->camera.Zoom(follow ? 0.5f : 2.0f);
	renderer->camera.Pan(vec2(0, 0));
}

void onInitialization() {
	renderer = new Renderer(new Shader());
	renderer->addObject(&biker);
	renderer->addObject(&ground);
	glViewport(0, 0, windowWidth, windowHeight); 	// Position and size of the photograph on screen
	glLineWidth(2.0f); // Width of lines in pixels
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.01f, 0.04f, 0.09f, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	renderer->render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case ' ': toggleFollowing(); break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		ground.addControlPoints(cX, cY);
		biker.reset();
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
	vec2 center = vec2(pX - windowWidth / 2.0f, pY - windowHeight / 2.0f);
	//renderer->camera.Pan(center * (1 / 1000.0f));

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static long lastTime = glutGet(GLUT_ELAPSED_TIME);
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the 
	float dT = (time - lastTime) * 0.001f;
	biker.move(dT);
	if (follow) renderer->camera.Pan(biker.getPosition() * 0.15f);
	lastTime = time;
	glutPostRedisplay();					// redraw the scene
}
