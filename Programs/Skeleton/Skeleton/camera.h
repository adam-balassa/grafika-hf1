#ifndef CAMERA_HEADER_INCLUDED
#define CAMERA_HEADER_INCLUDED

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
		void Pan(vec2 t) { wCenter = wCenter + t; }
	};
#endif



