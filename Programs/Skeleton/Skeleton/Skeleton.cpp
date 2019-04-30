//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

inline void printvec(const vec3& v) {
	printf("(%f %f %f)\n", v.x, v.y, v.z);
}

inline vec3 operator/(const vec3& v1, const vec3& v2) {
	return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

class Shader {
	const char *vertexSource = R"(
		#version 330
		precision highp float;

		uniform vec3 wLookAt, wRight, wUp;          // pos of eye

		layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
		out vec3 p;

		void main() {
			gl_Position = vec4(cCamWindowVertex, 0, 1);
			p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
		}
	)";
	char *fragmentSource = R"(
#version 330
precision highp float;

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

struct Light {
	vec3 direction;
	vec3 Le, La;
};

struct Sphere {
	vec3 center;
	float radius;
};

struct Ellipsoid {
	vec3 center;
	float radius, scaleX, scaleY;	
};

struct Plane{
	vec3 cp, normal;
};

struct Hit {
	float t;
	vec3 position, normal;
	int mat;	// material index
};

struct Ray {
	vec3 start, dir;
};

const int nMaxObjects = 50;

uniform vec3 wEye;
uniform Light light;
uniform Material materials[20];  // diffuse, specular, ambient ref
uniform int mirrorMaterial;

uniform int nEllipsoids;
uniform Ellipsoid ellipses[nMaxObjects];

uniform int nMirrors;
uniform Plane mirrors[nMaxObjects];

in  vec3 p;					// point on camera window corresponding to the pixel
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

mat4 scaleMatrix(float scaleX, float scaleY){
	return mat4(
		scaleX, 0, 		0, 0,
		0, 		scaleY, 0, 0,
		0, 		0, 		1, 0,
		0, 		0, 		0, 1
	);
}
mat4 invScaleMatrix(float scaleX, float scaleY){
	return mat4(
		1 / scaleX, 0, 		0, 0,
		0, 		1 / scaleY, 0, 0,
		0, 		0, 			1, 0,
		0, 		0, 			0, 1
	);
}


Hit intersect(const Sphere object, const Ray ray) {
	Hit hit;
	hit.t = -1;
	vec3 dist = ray.start - object.center;
	float a = dot(ray.dir, ray.dir);
	float b = dot(dist, ray.dir) * 2.0;
	float c = dot(dist, dist) - object.radius * object.radius;
	float discr = b * b - 4.0 * a * c;
	if (discr < 0) return hit;
	float sqrt_discr = sqrt(discr);
	float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
	float t2 = (-b - sqrt_discr) / 2.0 / a;
	if (t1 <= 0) return hit;
	hit.t = (t2 > 0) ? t2 : t1;
	hit.position = ray.start + ray.dir * hit.t;
	hit.normal = (hit.position - object.center) / object.radius;
	return hit;
}

Hit intersect(const Ellipsoid o, const Ray ray){
	Sphere sphere;
	mat4 scale = scaleMatrix(o.scaleX, o.scaleY);
	mat4 invscale = invScaleMatrix(o.scaleX, o.scaleY);
	mat4 transposeinvscale = transpose(invscale);

	sphere.center = vec3(vec4(o.center, 1) * scale);
	sphere.radius = o.radius;

	Ray trRay;
	trRay.dir = vec3(vec4(ray.dir, 1) * invscale);
	trRay.start = vec3(vec4(ray.start, 1) * invscale);
	Hit h = intersect(sphere, trRay);
	h.normal = vec3(vec4(h.normal, 1) * transposeinvscale);
	h.position = vec3(vec4(h.position, 1) * scale);
	return h;
}

Hit intersect(const Plane m, const Ray ray){
	Hit h;
	float epsilon = 0.0001;
	float p1 = dot(m.normal, ray.dir);
	vec3 x = ray.start - m.cp;
	float p2 = -dot(m.normal, x);
	float lambda = p2 / p1;
	if (lambda < 0) return h;
	vec3 p = ray.dir * lambda + ray.start;
	if(length(p) > 12.0) return h;

	h.t = lambda;
	h.normal = m.normal;
	h.position = p + h.normal * epsilon;
	h.mat = mirrorMaterial;
	return h;
}

Hit firstIntersect(Ray ray) {
	Hit bestHit;
	bestHit.t = -1;
	for (int o = 0; o < nEllipsoids; o++) {
		Hit hit = intersect(ellipses[o], ray); //  hit.t < 0 if no intersection
		hit.mat = o % 8 + 2;
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
	}
	for (int o = 0; o < nMirrors; o++) {
		Hit hit = intersect(mirrors[o], ray); //  hit.t < 0 if no intersection
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
	}
	if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
	return bestHit;
}

bool shadowIntersect(Ray ray) {	// for directional lights
	for (int o = 0; o < nEllipsoids; o++) if (intersect(ellipses[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
	for (int o = 0; o < nMirrors; o++) if (intersect(mirrors[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
	return false;
}

vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (vec3(1, 1, 1) - F0) * pow(1 - cosTheta, 5);
}

const float epsilon = 0.0001f;
const int maxdepth = 10;

vec3 trace(Ray ray) {
	vec3 weight = vec3(1, 1, 1);
	vec3 outRadiance = vec3(0, 0, 0);
	for (int d = 0; d < maxdepth; d++) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return weight * light.La;
		if (materials[hit.mat].rough == 1) {
			outRadiance += materials[hit.mat].ka * light.La;
			Ray shadowRay;
			shadowRay.start = hit.position + hit.normal * 5000.0;
			shadowRay.dir = light.direction * -1;
			float cosTheta = dot(hit.normal, light.direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance += light.Le * materials[hit.mat].kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light.direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance += light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
			}
			outRadiance = outRadiance * weight;
		}

		if (materials[hit.mat].reflective == 1) {
			weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
			ray.start = hit.position + hit.normal * epsilon;
			ray.dir = reflect(ray.dir, hit.normal);
		}
		else return outRadiance;
	}
}

void main() {
	Ray ray;
	ray.start = wEye;
	ray.dir = normalize(p - wEye);
	fragmentColor = vec4(trace(ray), 1);
}	
)";
	GPUProgram shaderProgram;
	//void readFragmentShader() {
	//	FILE * f = fopen(R"(C:\Users\balas\Documents\BME\grafika\grafika-hf1\Programs\Skeleton\Skeleton\shader.txt)", "r");

	//	// Determine file size
	//	fseek(f, 0, SEEK_END);
	//	size_t size = ftell(f);

	//	fragmentSource = new char[size];

	//	rewind(f);
	//	fread((void*)fragmentSource, sizeof(char), size, f);
	//	fragmentSource[size - 1] = '\0';
	//}
public:
	void init() {
		//readFragmentShader();
		shaderProgram.Create(vertexSource, fragmentSource, "fragmentColor");
		shaderProgram.Use();
	}

	unsigned int getId() {
		return shaderProgram.getId();
	}

};
// vertex shader in GLSL

bool operator<(const vec3& v1, const vec3& v2) {
	return length(v1) < length(v2);
}
bool operator>(const vec3& v1, const vec3& v2) {
	return length(v1) > length(v2);
}

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		F0 = vec3(0, 0, 0);
		rough = true;
		reflective = false;
	}
};

class SmoothMaterial : public Material {
	vec3 getReflectance(const vec3& n, const vec3& k) {
		return
			((n - vec3(1, 1, 1)) * (n - vec3(1, 1, 1)) + k * k) /
			((n + vec3(1, 1, 1)) * (n + vec3(1, 1, 1)) + k * k);
	}
public:
	SmoothMaterial(const vec3& n, const vec3& k) {
		F0 = getReflectance(n, k);
		rough = false;
		reflective = true;
	}
};

struct Sphere {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) { center = _center; radius = _radius; }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "objects[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "objects[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius); else printf("uniform %s cannot be set\n", buffer);
	}
};

struct Ellipsoid : Sphere{
	float scaleX, scaleY;
	vec3 v = vec3(1, 1, 0);
	Ellipsoid(const vec3& _center, float _radius, float scaleX, float scaleY): 
		Sphere(_center, _radius), scaleX(scaleX), scaleY(scaleY) {  }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "ellipses[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "ellipses[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius); else printf("uniform %s cannot be set\n", buffer);
		sprintf(buffer, "ellipses[%d].scaleX", o);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, scaleX); else printf("uniform %s cannot be set\n", buffer);
		sprintf(buffer, "ellipses[%d].scaleY", o);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, scaleY); else printf("uniform %s cannot be set\n", buffer);
	}
};

struct Mirror {
	vec3 point;
	vec3 normal;
	Mirror(vec3 point, vec3 normal): point(point), normal(normal) { }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "mirrors[%d].cp", o);
		point.SetUniform(shaderProg, buffer);		
		sprintf(buffer, "mirrors[%d].normal", o);
		normal.SetUniform(shaderProg, buffer);

	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Ellipsoid *> ellipsoids;
	std::vector<Mirror*> mirrors;
	std::vector<Light *> lights;
	Camera camera;
	std::vector<Material *> materials;
	bool goldMirror = true;
	Shader* shader;

	void initMirrors(unsigned int numOfMirrors) {
		for (Mirror* m : mirrors) delete m;
		mirrors.clear();
		const float radius = 0.7f;
		const vec3 center(0, 0, 0);
		const float angle = 2 * M_PI / numOfMirrors; 
		for (int n = 0; n < numOfMirrors; ++n) {
			vec3 point = vec3(sinf(n * angle) * radius, cosf(n * angle) * radius, 0);
			mirrors.push_back(new Mirror(point, normalize(point) * -1));
		}

		SetUniform();
	}

public:
	void build() {
		shader = new Shader();
		shader->init();
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(1, 1, 0.7f), vec3(1.5f, 1.5f, 1.5f), vec3(0.5, 0.5, 0.5)));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(0.3f, 0.3f, 0.3f);
		for (int i = 0; i < 5; i++) ellipsoids.push_back(new Ellipsoid(vec3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5 - 6.0f), rnd() * 0.1 + 0.05f, 1.0f, 1.4f));

		materials.push_back(new SmoothMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f)));
		materials.push_back(new SmoothMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f)));

		vec3 c1(0.8f, 0.2f, 0.1f),
			c2(0.1f, 0.5f, 0.3f),
			c3(0.9f, 0.2f, 0.8f), c4(0.2f, 0.4f, 0.9f), c5(0.6f, 0.5f, 0.1f), c6(0.4f, 0.8f, 0.3f), c7(0.1f, 0.3f, 0.95f);
		materials.push_back(new RoughMaterial(kd, ks, 30));
		materials.push_back(new RoughMaterial(c1, ks, 20));
		materials.push_back(new RoughMaterial(c2, ks, 20));
		materials.push_back(new RoughMaterial(c3, ks, 20));
		materials.push_back(new RoughMaterial(c4, ks, 20));
		materials.push_back(new RoughMaterial(c5, ks, 20));
		materials.push_back(new RoughMaterial(c6, ks, 20));
		materials.push_back(new RoughMaterial(c7, ks, 20));
		initMirrors(3);
	}

	void addMirror() {
		initMirrors(mirrors.size() + 1);
	}

	void setMirrorMaterial(bool gold) {
		goldMirror = gold;
		SetUniform();
	}

	
	void SetUniform() {
		unsigned int shaderProg = shader->getId();
		int location = glGetUniformLocation(shaderProg, "nEllipsoids");
		if (location >= 0) glUniform1i(location, ellipsoids.size()); else printf("uniform nEllipsoids cannot be set\n");
		for (int o = 0; o < ellipsoids.size(); o++) ellipsoids[o]->SetUniform(shaderProg, o);

		location = glGetUniformLocation(shaderProg, "nMirrors");
		if (location >= 0) glUniform1i(location, mirrors.size() + 1); else printf("uniform nMirrors cannot be set\n");
		for (int o = 0; o < mirrors.size(); ++o) mirrors[o]->SetUniform(shaderProg, o);

		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);
		for (int mat = 0; mat < materials.size(); mat++) materials[mat]->SetUniform(shaderProg, mat);

		location = glGetUniformLocation(shaderProg, "mirrorMaterial");
		if (location >= 0) glUniform1i(location, goldMirror ? 0 : 1); else printf("uniform mirrorMaterial cannot be set\n");
	}
	void Animate(float dt) {
		for (Ellipsoid* ellipsoid : ellipsoids) {
			ellipsoid->center = ellipsoid->center + ellipsoid->v * dt;
			ellipsoid->v = normalize(ellipsoid->v + vec3(rnd() - 0.5f, rnd() - 0.5f, 0) * dt * 1000.0f);
		}
	}
};

Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	srand(glutGet(GLUT_ELAPSED_TIME));
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.Create();

}

// Window has become invalid: Redraw
void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.SetUniform();
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a')
		scene.addMirror(); 
	else if (key == 'g')
		scene.setMirrorMaterial(true);
	else if (key == 's')
		scene.setMirrorMaterial(false);
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
	float dT = (time - lastTime) * 0.0001f;
	scene.Animate(dT);
	glutPostRedisplay();
	lastTime = time;
}