//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"
#define max(a, b) a > b ? a : b

const int tessellationLevel = 20;
class LadyBird;


struct quat {
	float r, i, j, k;
	quat(float i, float j, float k, float r): r(r), i(i), j(j), k(k){}
	quat(vec4 v) {
		r = v.w;
		i = v.x;
		j = v.y;
		k = v.z;
	}
	quat operator*(const quat& q) const {
		vec3 d1(i, j, k), d2(q.i, q.j, q.k);
		vec3 a(d2 * r + d1 * q.r + cross(d1, d2));
		return quat(a.x, a.y, a.z, r * q.r - dot(d1, d2));
	}
	static vec3 rotate(vec3 u, quat q) {
		quat qinv(-q.i, -q.j, -q.k, q.r);
		quat qr = q * quat(u.x, u.y, u.z, 0) * qinv;
		return vec3(qr.i, qr.j, qr.k);
	}

};

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
	LadyBird* ladyBird;
	float distance = 5.0f;
public:
	Camera(LadyBird* l): ladyBird(l) {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float t);
	void toggleDistance() {
		distance = distance > 3.0f ? 1.2f : 5.0f;
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
	Material(){}
	Material(vec3 kd, vec3 ks, vec3 ka, float shininess): kd(kd), ks(ks), ka(ka), shininess(shininess){}
	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	
		const quat q(
			cosf(t / 4.0f),
			sinf(t / 4.0f) * cosf(t) * 0.5f,
			sinf(t / 4.0f) * sinf(t) * 0.5f,
			sinf(t / 4.0f) * sqrtf(0.75f)
		);

		vec3 a = quat::rotate(quat::rotate(vec3(wLightPos.x, wLightPos.y, wLightPos.z), q), q);
		wLightPos = vec4(a.x, a.y, a.z, wLightPos.w);
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

//---------------------------
struct CheckerBoardTexture : public Texture {
	//---------------------------
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		const vec3 yellow(1, 1, 0), blue(0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

class LadybirdTexture : public Texture {
	//---------------------------
	float random(int i) {
		return (float)rand() / RAND_MAX;
	}
	std::vector<vec2> dots;
	const float radius = 10.0f;
	vec3 getPixel(int x, int y) {
		const vec3 red(1, 0, 0), black(0, 0, 0), white(1, 1, 1);
		if (length(vec2(320, 130) - vec2(x, y)) < 15.0f) return white;
		if (length(vec2(320, 70) - vec2(x, y)) < 15.0f) return white;
		if (abs(y - 100.0f) < 1.2f) return black;
		for (vec2 dot : dots)
			if (length(dot - vec2(x, y)) < radius)
				return black;
		if (length(vec2(355, 110) - vec2(x, y)) < 11.0f) return white;
		if (length(vec2(355, 90) - vec2(x, y)) < 11.0f) return white;
		return red;
	}
public:
	LadybirdTexture(const int width = 400, const int height = 200) : Texture() {
		dots.push_back(vec2(45, 70));
		dots.push_back(vec2(45, 130));
		dots.push_back(vec2(380, 60));
		dots.push_back(vec2(380, 150));
		dots.push_back(vec2(8, 80));
		dots.push_back(vec2(8, 120));
		dots.push_back(vec2(408, 80));
		dots.push_back(vec2(408, 120));
		dots.push_back(vec2(370, 100));
		dots.push_back(vec2(373, 100));
		dots.push_back(vec2(370, 105));
		dots.push_back(vec2(370, 95));
		for (int j = 0; j < 6; ++j) for (int i = 0; i < 15; ++i) dots.push_back(vec2(320 + 6.0 * (j - abs(i - 5.0f) * 0.4f), 40 + 12 * i));
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = getPixel(x, y);
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//--------------------------
public:
	virtual void Bind(RenderState state) = 0;
};

//---------------------------
class GouraudShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.lights[0].wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal, du, dv;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vd.normal = vec3(cosf(u * 2.0f * M_PI) * sinf(v*M_PI), sinf(u * 2.0f * M_PI) * sinf(v*M_PI), cosf(v*M_PI));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Torus : public ParamSurface {
	//---------------------------
	const float R = 1, r = 0.5;

	vec3 Point(float u, float v, float rr) {
		float ur = u * 2.0f * M_PI, vr = v * 2.0f * M_PI;
		float l = R + rr * cosf(ur);
		return vec3(l * cosf(vr), l * sinf(vr), rr * sinf(ur));
	}
public:
	Torus() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = Point(u, v, r);
		vd.normal = (vd.position - Point(u, v, 0)) * (1.0f / r);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Mobius : public ParamSurface {
	//---------------------------
	float R, w;
public:
	Mobius() { R = 1; w = 0.5; Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * M_PI, V = (v - 0.5) * w;
		Clifford x = (Cos(T(U)) * V + R) * Cos(T(U) * 2);
		Clifford y = (Cos(T(U)) * V + R) * Sin(T(U) * 2);
		Clifford z = Sin(T(U)) * V;
		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdU(x.d, y.d, z.d);
		vec3 drdV(cos(U)*cos(2 * U), cos(U)*sin(2 * U), sin(U));
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class HalfSphere : public Sphere {
	//---------------------------
	float R, w;
public:
	HalfSphere() { R = 1; w = 0.5; Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd = Sphere::GenVertexData(u, v);
		if (vd.position.x < 0) {
			vd.position.x = 0;
			vd.normal = vec3(1, 0, 0);
		}
		return vd;
	}
};

class Klein : public ParamSurface {
	//---------------------------
public:
	Klein() {Create(); }



	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2.0f * M_PI, V = v * 2.0f * M_PI;
		Clifford	a = Cos(T(U)) * (Sin(T(U)) + 1) * 6, 
					b = Sin(T(U)) * 16,
					c = ((Cos(T(U)) * (-0.5f)) + 1) * 4;

		bool cond = U > M_PI && U < 2 * M_PI + 0.001f;
		Clifford	x = cond ? (a + c * cosf(V + M_PI)) : (a + c * Cos(T(U)) * cosf(V)),
					y = cond ? b : (b + c * Sin(T(U)) * cosf(V)),
					z = c * sinf(V);

		vd.position = vec3(x.f, y.f, z.f) * 0.17f;
		vd.du = vec3(x.d, y.d, z.d);
		vd.dv = vec3(
			cond ? c.f * (-1) * sinf(V + M_PI) : c.f * cosf(U) * (-1) * sinf(V),
			cond ? 0 : c.f * sinf(U) * (-1) * sinf(V),
			c.f * cosf(V)
		);
		vd.normal = cross(vd.du, vd.dv);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};
class Dini : public ParamSurface {
	//---------------------------
public:
	Dini() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float a = 1.0f, b = 0.15f;
		float U = u * 4.0f * M_PI, V = max(0.05f, v);
		Clifford x = Sin(T(V)) * cosf(U) * a,
				 y = Sin(T(V)) * sinf(U) * a,
				 z = Cos(T(V)) + Log(Tan(T(V / 2))) + b * U;

		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdV(x.d, y.d, z.d);
		vec3 drdU(
			-sinf(U) * sinf(V) * a,
			cosf(U) * sinf(V) * a,
			b
		);
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Plane : public ParamSurface {
public:
	Plane() { Create(); }
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vec3((u - 0.5f) * 4, (v - 0.5f) * 4);
		vd.normal = vec3(0, 0, -1);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};
//---------------------------
struct Object {
	//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { };
	virtual ~Object() {
		delete shader;
		delete texture;
		delete geometry;
		delete material;
	}
};

struct Planet : public Object {
	Planet(): Object(
		new PhongShader(), 
		new Material(vec3(0.5f, 0.5f, 0.5f), vec3(5, 5, 5), vec3(0.5f,0.5f,0.5f), 200), 
		new CheckerBoardTexture(30, 30),
		new Klein()) {
		rotationAngle = 0;
	}
	
	void getPosition(const float u, const float v, vec3& pos, vec3& norm, vec3& du, vec3& dv) const {
		VertexData vd = ((ParamSurface*)this->geometry)->GenVertexData(u, v);
		norm = vd.normal;
		pos = vd.position + translation;
		du = vd.du;
		dv = vd.dv;
	}
};

class LadyBird : public Object {
	float angle = 0;
	float velocity = 0.8f;
	vec2 position = vec2(0.2f, 0);
	Planet* planet;
	mat4 m, mInv;
	float x = 0;

public:
	LadyBird(Planet* p): Object(
		new NPRShader(),
		new Material(vec3(0.6f, 0.4f, 0.2f), vec3(4, 4, 4), vec3(0.1f, 0.1f, 0.1f), 100),
		new LadybirdTexture(),
		new HalfSphere()
	), planet(p){
	}

	void turn(const float dAngle) {
		angle += dAngle;
	}
	void Animate(float tStart, float tEnd) {
		static float lastT = tStart;
		const float dT = lastT - tEnd;

		vec3 du, dv, normal;
		planet->getPosition(position.x, position.y, translation, normal, du, dv);

		vec2 derivative(length(du), length(dv));
		derivative = normalize(derivative);
		vec2 s = vec2(1 / length(du), 1 / length(dv)) * dT * velocity;

		const vec2 d(sinf(angle) * s.x, cosf(angle) * s.y);
		position = position + d;

		vec3 dr = du * sinf(angle) * s.x + dv * cosf(angle) * s.y;
		vec3 j = normalize(-dr), i = normalize(normal);
		vec3 k = cross(j, i);
		m = mat4(
			i.x, i.y, i.z, 0,
			j.x, j.y, j.z, 0,
			k.x, k.y, k.z, 0,
			0, 0, 0, 1
		);

		mInv = mat4(
			i.x, j.x, k.x, 0,
			i.y, j.y, k.y, 0,
			i.z, j.z, k.z, 0,
			0, 0, 0, 1
		);

		lastT = tEnd;
	}

	void Draw(RenderState state) {
		vec3 du, dv, normal;
		planet->getPosition(position.x, position.y, translation, normal, du, dv);
		vec3 dr = du * sinf(angle) + dv * cosf(angle);
		

		state.M = ScaleMatrix(scale) * m * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * mInv * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void getPosition(vec3& pos, vec3& norm) {
		vec3 du, dv;
		planet->getPosition(position.x, position.y, pos, norm, du, dv);
	}
};

void Camera::Animate(float t) {
	vec3 pos, norm;
	ladyBird->getPosition(pos, norm);
	wEye = pos + normalize(norm) * distance;
	wLookat = pos;
}

class Tree : public Object {

public:
	Tree(Planet* p, vec2 position) : Object(
		new PhongShader(),
		new Material(vec3(0.7f, 0.2f, 0.1f), vec3(4, 4, 4), vec3(0.1f, 0.1f, 0.1f), 100),
		new CheckerBoardTexture(10, 10),
		new Dini()
	) {
		vec3 norm, du, dv;
		p->getPosition(position.x, position.y, translation, norm, du, dv);
		norm = normalize(norm);
		rotationAxis = cross(vec3(0, 0, 1), norm);
		const float cosTheta = dot(vec3(0, 0, 1), norm);
		rotationAngle = acosf(cosTheta);
		translation = translation + norm * 0.5f;
		scale = vec3(1, 1, 1) * 0.2f;
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object *> objects;
public:
	Camera* camera; // 3D camera
	LadyBird* ladyBird;
	std::vector<Light> lights;

	void Build() {
		Planet * planet = new Planet();
		planet->rotationAxis = vec3(1, 1, -1);
		objects.push_back(planet);

		ladyBird = new LadyBird(planet);
		ladyBird->scale = vec3(1, 1.2f, 1) * 0.1f;
		objects.push_back(ladyBird);

		Tree* tree = new Tree(planet, vec2(0.3f, 0.9f));
		objects.push_back(tree);

		Tree* tree2 = new Tree(planet, vec2(0.1f, 0.4f));
		objects.push_back(tree2);
		Tree* tree3 = new Tree(planet, vec2(0.2f, 0.8f));
		objects.push_back(tree3);
		Tree* tree4 = new Tree(planet, vec2(0.58f, 0.1f));
		objects.push_back(tree4);
		Tree* tree5 = new Tree(planet, vec2(0.7f, 0.7f));
		objects.push_back(tree5);
		Tree* tree6 = new Tree(planet, vec2(0.55f, 0.85f));
		objects.push_back(tree6);
		Tree* tree7 = new Tree(planet, vec2(0.38f, 0.2f));
		objects.push_back(tree7);

		// Camera
		camera = new Camera(ladyBird);
		camera->wEye = vec3(0, 0, 6);
		camera->wLookat = vec3(0, 0, 0);
		camera->wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.4, 0.4, 0.4);
		lights[0].Le = vec3(3, 3, 3);

	}
	void Render() {
		RenderState state;
		state.wEye = camera->wEye;
		state.V = camera->V();
		state.P = camera->P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera->Animate(tend);
		for (int i = 0; i < lights.size(); i++) { lights[i].Animate(tend - tstart); }
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
	~Scene() {
		for (Object* o : objects) delete o;
		delete camera;
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	switch (key)
	{
	case 'a':
		scene.ladyBird->turn(M_PI / 4);
		break;
	case 's':
		scene.ladyBird->turn(-M_PI / 4);
		break;
	case ' ':
		scene.camera->toggleDistance();
		break;
	default:
		break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}