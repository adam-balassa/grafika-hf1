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

inline void printVec3(const vec3& v) {
	printf("(%f %f %f)\n", v.x, v.y, v.z);
}

inline vec4 vec3ToVec4(const vec3& v) { return vec4(v.x, v.y, v.z, 0); }

inline vec3 vec4ToVec3(const vec4& v) { return vec3(v.x, v.y, v.z); }

inline float rnd() { return (float)rand() / RAND_MAX; }

inline bool operator==(const vec3& v1, const vec3& v2) {
	const float epsilon = 0.0001f;
	return abs(length(v1 - v2)) < epsilon;
}

inline vec3 rndVec3() {
	return vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f);
}

const float PI = (float)M_PI;

inline vec2  m(const vec2& v1, const vec2& v2) {
	return (v2 - v1) * (1 / (v2.x - v1.x));
}
inline float  m2(const vec2& v1, const vec2& v2) {
	return (v2.y - v1.y) * (1 / (v2.x - v1.x));
}

class Shader {
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

	GPUProgram textureProgram;
public:
	void initializeShaders() {
		textureProgram.Create(textureVertexShader, textureFragmentShader, "fragmentColor");
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

	void setImage(const std::vector<vec3>& image, const unsigned int width, const unsigned int height) {
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

struct Material {
	vec3 ka, kd, ks;
	float  shininess, reflectivity;
	Material(vec3 _kd, vec3 _ks, float _shininess, float reflectivity = 0) :
		ka(_kd * M_PI),
		kd(_kd),
		ks(_ks),
		shininess(_shininess),
		reflectivity(reflectivity)
	{}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir, shine = vec3(1.0f, 1.0f, 1.0f);
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Transformation {
	bool rotated = false, translated = false, scaled = false;

	mat4 unitM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 scaleM() const {
		return mat4(
			scaleX, 0, 0, 0,
			0, scaleY, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 invscaleM() const {
		return mat4(
			1 / scaleX, 0, 0, 0,
			0, 1 / scaleY, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 translateM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			translation.x, translation.y, 0, 1);
	}

	mat4 invtranslateM() const {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
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
	float scaleX = 1.0f, scaleY = 1.0f;

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
};

class GameObject {
protected:
	Material *material;
	Transformation transformation;
public:
	virtual Hit intersect(const Ray& ray) const = 0;
	virtual ~GameObject() {
		if (material != nullptr)
			delete material;
	}
};

struct Sphere : public GameObject {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}
	virtual Hit intersect(const Ray& ray) const {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;

		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Sphere {
	Ellipsoid(const vec3& _center, float _radius, Material* _material) : Sphere(_center, _radius, _material) {
		transformation.scale(1.0f, 1.5f);
		center = vec4ToVec3(vec3ToVec4(center) * transformation.getInversTransformationMatrix());
	}

	Hit intersect(const Ray& ray) const {
		vec4 transformedDir = vec3ToVec4(ray.dir) * transformation.getInversTransformationMatrix();
		vec4 transformedStart = vec3ToVec4(ray.start) * transformation.getInversTransformationMatrix();
		Ray newRay = Ray(vec4ToVec3(transformedStart), vec4ToVec3(transformedDir));
		Hit h = Sphere::intersect(newRay);
		h.normal = vec4ToVec3(vec3ToVec4(h.normal) * transformation.getInversTransformationMatrix());
		h.position = vec4ToVec3(vec3ToVec4(h.position) * transformation.getTransformationMatrix());
		return h;
	}
};

struct MovingEllipsoid : public Ellipsoid {
	MovingEllipsoid(Material *material) : Ellipsoid(rndVec3(), rnd() * 0.1f, material) {}

	void move() {
		center = center + rndVec3() * 0.1f;
	}
};

struct Mirror : GameObject {
	vec3 point, norm;

public:
	Mirror(vec3 point, vec3 norm) :
		point(point),
		norm(norm)
	{
		material = new Material(vec3(0, 0, 0), vec3(0, 0, 0), 0, 0.9f);
	}

	Hit intersect(const Ray& ray) const {
		const float epsilon = 0.0001f;
		float p1 = dot(norm, ray.dir);
		const vec3 x = ray.start - point;
		float p2 = -dot(norm, x);
		const float lambda = p2 / p1;
		if (lambda < 0) return Hit();
		const vec3 p = ray.dir * lambda + ray.start;

		Hit h = Hit();
		h.t = length(ray.dir * lambda);
		h.material = material;
		h.normal = norm;
		h.position = p;
		return h;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};


const float epsilon = 0.0001f;

class Scene {
	std::vector<GameObject *> objects;
	std::vector<Light *> lights;
	TextureMesh* texture;
	Shader* shader;
	std::vector<vec3> image;
	Camera camera;
	vec3 La;

	unsigned int reflections = 0;

	vec3 directLight(const Hit& hit, const Ray& ray) {
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}

	vec3 countReflectionDir(const vec3& v, const vec3& n) {
		vec3 norm = normalize(n);
		/*printVec3(v);
		printVec3(n);
		printVec3(v - (n * dot(v, n) * 2));
		printf("\n\n");*/
		return v - (n * dot(v, n) * 2);
	}

	vec3 reflect(const Hit& hit, const Ray& ray) {
		if (++reflections >= 16) return vec3(0, 0, 0);
		vec3 dir = countReflectionDir(ray.dir, hit.normal);
		Ray reflectedRay(hit.position + dir * epsilon, dir);
		return trace(reflectedRay) * hit.material->reflectivity;
	}

public:
	void build() {
		image = std::vector<vec3>(windowWidth * windowHeight);
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.7f, 0.2f, 0.1f), ks(0.1f, 0.1f, 0.1f);
		Material * material = new Material(kd, ks, 20);
		for (int i = 0; i < 4; i++) addObject(new Sphere(vec3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5), rnd() * 0.1, material));

		vec3 kd2(0.1f, 0.2f, 0.8f), ks2(0.1f, 0.1f, 0.1f);
		Material * material2 = new Material(kd2, ks2, 20);
		for (int i = 0; i < 4; i++) addObject(new Ellipsoid(vec3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5), rnd() * 0.1, material2));

		shader = new Shader();
		shader->initializeShaders();
		texture = new TextureMesh();
		texture->initialize();
	}

	void addObject(GameObject* o) {
		objects.push_back(o);
	}



	void render() {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				reflections = 0;
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec3(color.x, color.y, color.z);
			}
		}
	}

	void draw() {
		texture->setImage(image, windowWidth, windowHeight);
		shader->bindTexture(texture->getTextureId());
		texture->draw();
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (GameObject * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (GameObject * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(const Ray& ray) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		if (hit.material->reflectivity < epsilon) return directLight(hit, ray);
		else return reflect(hit, ray);

	}
	~Scene() {
		for (GameObject *object : objects)
			delete object;
	}
};
Scene scene;
MovingEllipsoid* e1;
MovingEllipsoid* e2;
MovingEllipsoid* e3;
Mirror* mirror;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	vec3 kd(0.1f, 0.5f, 0.1f), ks(0.1f, 0.1f, 0.1f);
	Material * material = new Material(kd, ks, 20.0f);
	e1 = new MovingEllipsoid(material);
	e2 = new MovingEllipsoid(material);
	e3 = new MovingEllipsoid(material);
	mirror = new Mirror(vec3(-1.0f, 0.0f, -3.0f), vec3(0.5f, 0.1f, 2.0f));
	scene.addObject(e1); scene.addObject(e2); scene.addObject(e3);
	scene.addObject(mirror);
	scene.render();
}

// Window has become invalid: Redraw
void onDisplay() {
	scene.render();
	scene.draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static long lastTime = glutGet(GLUT_ELAPSED_TIME);
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the 
	srand(time + lastTime);
	float dT = (time - lastTime) * 0.001f;
	e1->move();
	e2->move();
	e3->move();
	lastTime = time;
	glutPostRedisplay();
}