#ifndef BIKER_HEADER_INCLUDED
#define BIKER_HEADER_INCLUDED

#include <math.h>

const float maxWidth = 10.0f;

class BikerPart : public GraphicsObject {
public:
	virtual void create() {
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

	virtual void scale(const float scale) = 0;
	virtual void replace(const vec2& pos) = 0;
	virtual void turn() = 0;
};

class StaticBody : public BikerPart {
	Circle head;
	const vec2 headPos = vec2(0, 3.8);
public:
	StaticBody(const vec3& color) : head(Circle(0.8, color, true)) {
		this->color = color;
	}

	void scale(const float scale) {
		transformations.scale(scale);
		head.scale(scale);
	}

	void replace(const vec2& v) {
		transformations.translate(v);
		head.replace(v);
	}

	void turn() {
		transformations.turn();
	}

	void create() {
		BikerPart::create();
		world->addObject(&head);
		head.setCenter(headPos);
		addPoint(vec2(0, 0));
		addPoint(vec2(0, 3));

		addPoint(vec2(0, 2.6));
		addPoint(vec2(-0.4, 1));

		addPoint(vec2(-0.4, 1));
		addPoint(vec2(0.8, -0.3));
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_STATIC_DRAW);
	}

	void draw() {
		mat4 MVPTransform = M();
		MVPTransform.SetUniform(world->programId(), "MVP");
		glBindVertexArray(vao);
		glDrawArrays(GL_LINES, 0, points.size() / 5);
	}	
};

class Leg : public BikerPart{
	float phi = 0;
	const float pedalRadius = 0.6; 
	const float thigh = 2;
	const vec2 pedalPos;
	bool left = false;

	vec2 knee(const vec2& footPos, const vec2& bottomPos) {
		vec2 m = vec2((footPos.x + bottomPos.x) / 2, (footPos.y + bottomPos.y) / 2);
		vec2 norm = normalize(vec2(-footPos.y + bottomPos.y, footPos.x - bottomPos.x));
		float c = length(footPos - bottomPos);
		float d = sqrt(thigh * thigh - c * c / 4);
		return m + norm * d;
	}

	vec2 foot() {
		return vec2(pedalRadius * sin(phi) + pedalPos.x, pedalRadius * cos(phi) + pedalPos.y);
	}

	void setPoints() {
		const vec2 bottomPos = vec2(0, 0);
		const vec2 footPos = foot();
		const vec2 kneePos = knee(footPos, bottomPos);

		points.clear();
		addPoint(bottomPos);
		addPoint(kneePos);
		addPoint(footPos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_DYNAMIC_DRAW);
	}
public:

	Leg(const vec3& color, const vec2& pedalPos): pedalPos(pedalPos) {
		this->color = color;
	}
	Leg(const vec3& color, const vec2& pedalPos, bool left) : pedalPos(pedalPos), left(left) {
		this->color = color;
	}
	void ride(const float dPhi) {
		phi = dPhi + (left ? 0.0f : (const float)M_PI);
		setPoints();
	}
	void scale(const float scale) {
		transformations.scale(scale);
	}

	void replace(const vec2& v) {
		transformations.translate(v);
	}

	void turn() {
		transformations.turn();
	}

	void create() {
		BikerPart::create();
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_STATIC_DRAW);
	}

	void draw() {
		GraphicsObject::draw();
		glDrawArrays(GL_LINE_STRIP, 0, points.size() / 5);
	}
};

class Wheel : public BikerPart {
	float phi = 0;
	const float radius = 1.3;
	float scaleRatio = 1;
	const unsigned int spokes = 13;
	const vec2 center;
	short direction = -1;
	Circle wheelCircle;

	void initializeWheel() {
		const float deltaPhi = 2 * M_PI / spokes;
		for (unsigned int i = 0; i < spokes; ++i) {
			addPoint(vec2(0, 0));
			addPoint(vec2(sin(deltaPhi * i) * radius, cos(deltaPhi * i) * radius));
		}
		transformations.translate(center);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_DYNAMIC_DRAW);
	}

public:
	Wheel(const vec2& center) : center(center), wheelCircle(Circle(radius, vec3(0.2,0.2,0.2))) {
		this->color = vec3(0.35, 0.35, 0.35);
	}

	void rotate(const float dPhi) {
		phi = dPhi;	
		transformations.rotate(direction * phi / (1));
	}

	void scale(const float scale) {
		transformations.scale(scale);
		wheelCircle.scale(scale);
		transformations.translate(transformations.translation * scale);
		scaleRatio = scale;
	}

	float move(const float s) {
		const float r = radius * scaleRatio;
		const float pi = M_PI;
		const float rotation = phi + s / r;
		rotate(rotation);
		return rotation;
	}

	void replace(const vec2& v) {
		transformations.translate(transformations.translation + v);
		wheelCircle.replace(v);
	}

	void turn() {
		direction *= -1;
	}

	void create() {
		BikerPart::create();
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_STATIC_DRAW);
		world->addObject(&wheelCircle);
		wheelCircle.setCenter(center);
		initializeWheel();
	}

	void draw() {
		GraphicsObject::draw();
		glDrawArrays(GL_LINES, 0, points.size() / 5);
	}

	vec2 getCenterFromPosition(const vec2& realPos, const float derivative) const {
		const vec2 norm = normalize(vec2(1, -derivative));
		return norm * radius + realPos;
	}

};

class Seat : public BikerPart {
	const vec2 wheelPos;

	void initializePoints() {
		addPoint(vec2(0, -0.3));     addPoint(wheelPos);
		addPoint(vec2(-0.7, -0.3));  addPoint(vec2(0.7, -0.3));
		addPoint(vec2(-1, 0.2));  addPoint(vec2(-0.7, -0.3));
	}

public:
	Seat(const vec3& color, const vec2& wheelPos): wheelPos(wheelPos) {
		this->color = color;
	}
	void create() {
		BikerPart::create();
		initializePoints();

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), &points[0], GL_STATIC_DRAW);
	}
	
	void replace(const vec2& v) {
		transformations.translate(v);
	}

	void scale(const float scale) {
		transformations.scale(scale);
	}

	void turn() {
		transformations.turn();
	}

	void draw() {
		mat4 MVPTransform = M();
		MVPTransform.SetUniform(world->programId(), "MVP");
		glBindVertexArray(vao);
		glDrawArrays(GL_LINES, 0, points.size() / 5);
	}
	

};

class Biker : public GraphicsObject {

	StaticBody* body;
	Leg* leg;
	Leg* leftLeg;
	Seat* seat;
	Wheel* wheel;

	const Kochanek& mountain;

	const vec2 wheelPos = vec2(0, -3.2);
	vec2 velocity = vec2(1, 0);
	vec2 position = vec2(0, 0);
	vec2 realPosition = vec2(0, 0);
	float scaleRatio = 1;
	bool direction = true;
public:
	Biker(const vec3& color, const Kochanek& mountain): mountain(mountain) {
		this->color = color; 
		body = new StaticBody(color);
		leg = new Leg(color, wheelPos);
		leftLeg = new Leg(color, wheelPos, true);
		wheel = new Wheel(wheelPos);
		seat = new Seat(vec3(0.8, 0.8, 0.8), wheelPos);
	}

	vec2 a()const {
		const float rho = 0.05f;
		const float v = length(velocity);
		const vec2 air = -normalize(velocity) * v * v * rho;
		const vec2 g = vec2(1, mountain.getDerivative(realPosition.x)) * 0.4;
		return g + air;
	}

	/*void move(const float t) {
		velocity = velocity + a();
		const float distLength = length(velocity) * t;
		const float dPhi = wheel->move(distLength);
		leg->ride(dPhi * 2);
		leftLeg->ride(dPhi * 2);
		const float slope = mountain.getDerivative(realPosition.x);
		const vec2 dist = vec2(1, slope) * distLength;
		position = position + dist;
		realPosition = wheel->getRealPosition(position, slope);
		replace();
	}*/

	void move(const float t) {
		//velocity = velocity + a();
		const float distLength = length(velocity) * t;
		const float dPhi = wheel->move(distLength);
		leg->ride(dPhi * 3);
		leftLeg->ride(dPhi * 3);
		/*const vec3 newPositionData = mountain.getPosition(realPosition.x, distLength, direction);
		realPosition = vec2(newPositionData.x, newPositionData.y);
		const float derivative = newPositionData.z;
		position = wheel->getCenterFromPosition(realPosition, derivative) - wheelPos;*/
		//printf("(%f %f) (%f %f)\n", position.x, position.y, realPosition.x, realPosition.y);
		//replace();
		//if (position.x > maxWidth || position.x < -maxWidth) turn();
	}

	void replace() {
		leg->replace(position);
		leftLeg->replace(position);
		seat->replace(position);
		body->replace(position);
		wheel->replace(position);
	}

	void scale(const float scale) {
		scaleRatio = scale;
		leg->scale(scale);
		leftLeg->scale(scale);
		seat->scale(scale);
		body->scale(scale);
		wheel->scale(scale);
	}

	void turn() {
		direction = !direction;
		leg->turn();
		leftLeg->turn();
		seat->turn();
		body->turn();
		wheel->turn();
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

		world->addObject(leftLeg);
		world->addObject(wheel);
		world->addObject(seat);
		world->addObject(body);
		world->addObject(leg);
		scale(0.4f);
	}

	void draw()const {
		//GraphicsObject::draw();
	}

	virtual ~Biker(){
		delete body;
	}
};

#endif