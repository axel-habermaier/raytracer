//---------------------------------------------------------------------------------------------------------------------------------
// Includes and usings
//---------------------------------------------------------------------------------------------------------------------------------
#include "Raytracer.cu.h"
#include <helper_math.h>

//---------------------------------------------------------------------------------------------------------------------------------
// Forward declarations 
//---------------------------------------------------------------------------------------------------------------------------------
__device__ bool calculateTriangleIntersection(const Ray& r, const Triangle& t, float& d, const float minD, const float maxD);
__device__ bool calculateSphereIntersection(const Ray& r, const Sphere& s, float& t, const float minD, const float maxD);
inline __device__ unsigned int rgbToInt(const float4& c);
__device__ bool calculateClosestIntersection(const Ray& ray, Intersection& intersection, const float minD, const float maxD,
	const bool showLights);
__device__ float4 calculateLighting(const Ray& ray, const Intersection& intersection);
__device__ float4 calculateRayColor(const Ray& ray);
__device__ void raytrace(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const int x, const int y,
	const float3& camPos, const float3& camRotMat1, const float3& camRotMat2, const float3& camRotMat3);
__global__ void raytraceGpu(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const Triangle* const triangles, const unsigned int numTriangles,
	const Sphere* const spheres, const unsigned int numSpheres,
	const Light* const lights, const unsigned int numLights, const bool showLights,
	const float3 camPos, const float3 camRotMat1, const float3 camRotMat2, const float3 camRotMat3);
void raytraceScene(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const Triangle* const triangles, const unsigned int numTriangles,
	const Sphere* const spheres, const unsigned int numSpheres,
	const Light* const lights, const unsigned int numLights, const bool showLights,
	const float3 camPos, const float3 camRotMat1, const float3 camRotMat2, const float3 camRotMat3);

__shared__ TriangleList _triangles;
__shared__ SphereList _spheres;
__shared__ LightList _lights;


//---------------------------------------------------------------------------------------------------------------------------------
// Implementation of the Möller-Trumbore algorithm for ray-triangle intersections.
// See paper for further details. However, only the distance to the intersection point is returned.
//---------------------------------------------------------------------------------------------------------------------------------
__device__ bool calculateTriangleIntersection(const Ray& r, const Triangle& t, float& d, const float minD, const float maxD)
{
	float3 tvec, pvec, qvec;
	float det, u, v;
	float3 tv = make_float3(t.vertex.x, t.vertex.y, t.vertex.z);
	float3 te1 = make_float3(t.edge1.x, t.edge1.y, t.edge1.z);
	float3 te2 = make_float3(t.edge2.x, t.edge2.y, t.edge2.z);

	pvec = cross(r.dir, te2);
	det = dot(te1, pvec);

	// Only the backward-face culling version is implemented.
	if (det < EPSILON)
	{
		return false;
	}

	tvec = r.origin - tv;
	u = dot(tvec, pvec);
	if (u < 0.0f || u > det)
	{
		return false;
	}

	qvec = cross(tvec, te1);
	v = dot(r.dir, qvec);
	if (v < 0.0f || u + v > det)
	{
		return false;
	}

	d = -dot(te2, qvec) / det;

	return d > minD && d < maxD;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Implementation of a geometric ray sphere intersection algorithm.
// See http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld013.htm for an explanation.
//---------------------------------------------------------------------------------------------------------------------------------
__device__ bool calculateSphereIntersection(const Ray& r, const Sphere& s, float& t, const float minD, const float maxD)
{
	float3 l = r.origin - make_float3(s.pos.x, s.pos.y, s.pos.z);
	float tca = dot(l, r.dir);

	// Facing away from sphere
	if (tca < 0.0f)
	{
		return false;
	}

	float ds = dot(l, l) - tca * tca;
	float rs = s.params.x * s.params.x;

	// d > r -> no intersection
	if (ds > rs)
	{
		return false;
	}

	t = tca - sqrtf(rs - ds);
	return t > minD && t < maxD;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Converts a float4 into an unsigned int.
// Float4s are used for color calculations. The final output colors, however, must be converted to unsigned integers. Alpha values
// are not supported.
//---------------------------------------------------------------------------------------------------------------------------------
inline __device__ unsigned int rgbToInt(const float4& c)
{
	// Clamp each component into [0; 255].
	float r = clamp(c.x * 255.0f, 0.0f, 255.0f);
	float g = clamp(c.y * 255.0f, 0.0f, 255.0f);
	float b = clamp(c.z * 255.0f, 0.0f, 255.0f);

	// Construct the unsigned int.
	return (int(r) << 16) | (int(g) << 8) | int(b);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Calculates the closest intersection for the given ray within the range of [minD; maxD].
//---------------------------------------------------------------------------------------------------------------------------------
__device__ bool calculateClosestIntersection(const Ray& ray, Intersection& intersection, const float minD, const float maxD,
	const bool showLights)
{
	const Triangle* t = 0;
	const Sphere* s = 0;
	float dt = 99999999.0f, ds = 99999999.0f;

	// Calculate the closest intersection with a triangle.
	for (unsigned int i = 0; i < _triangles.Num; ++i)
	{
		float d;
		if (calculateTriangleIntersection(ray, _triangles.List[i], d, minD, maxD))
		{
			if (d < dt)
			{
				dt = d;
				t = &_triangles.List[i];
			}
		}
	}

	// Calculate the closest intersection with a sphere.
	for (unsigned int i = 0; i < _spheres.Num; ++i)
	{
		float d;
		bool isLightSphere = _spheres.List[i].params.z == 1.0f;
		bool showSphere = !isLightSphere || (isLightSphere && showLights);
		if (showSphere && calculateSphereIntersection(ray, _spheres.List[i], d, minD, maxD))
		{
			if (d < ds)
			{
				ds = d;
				s = &_spheres.List[i];
			}
		}
	}

	// Check if there was an intersection and construct the intersection result structure.
	if (t != 0 && dt < ds)
	{
		intersection.point = ray.origin - dt * ray.dir;
		intersection.reflection = t->normal.w;
		intersection.color = t->color;
		intersection.normal = make_float3(t->normal.x, t->normal.y, t->normal.z);
		return true;
	}
	else if (s != 0)
	{
		intersection.point = ray.origin - ds * ray.dir;
		intersection.reflection = s->params.y;
		intersection.color = s->color;
		intersection.normal = normalize(intersection.point - make_float3(s->pos.x, s->pos.y, s->pos.z));
		return true;
	}

	return false;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Calculates the color of a lighted surface at an intersection using the Phong lighting equations.
//---------------------------------------------------------------------------------------------------------------------------------
__device__ float4 calculateLighting(const Ray& ray, const Intersection& intersection)
{
	// The ambient color.
	float4 color = make_float4(
		intersection.color.x * AMBIENT_INTENSITY,
		intersection.color.y * AMBIENT_INTENSITY,
		intersection.color.z * AMBIENT_INTENSITY,
		1.0f);

	// Check all lights.
	for (unsigned int i = 0; i < _lights.Num; ++i)
	{
		const Light& l = _lights.List[i];

		// The (normalized) vector (and its length) from the intersection point to the light source.
		float3 lightVector = make_float3(l.pos) - intersection.point;
		float3 nLightVector = normalize(lightVector);
		float len = length(lightVector);

		float d = dot(nLightVector, intersection.normal);

		// Check if the light is actually in front of the surface.
		if (d > 0.0f)
		{
			// Construct the shadow ray from the intersection point to the light.
			Ray shadowRay;
			shadowRay.origin = intersection.point;
			shadowRay.dir = -nLightVector;

			// Check if there are any intersections between the intersection point and the light. If so, the intersection
			// point lies in shadows and we can skip the rest of the lighting calculations for the current light.
			Intersection unused;
			if (!calculateClosestIntersection(shadowRay, unused, EPSILON, len, false))
			{
				// The attenuation is 1 - (distance / radius)^2.
				float attenuation = len / l.pos.w;
				attenuation *= attenuation;
				attenuation = clamp(1 - attenuation, 0.0f, 1.0f);

				// Calculate the light color, modified by the attenuation, the light angle and the intersection color.
				float4 lColor = l.color * attenuation;
				float4 dColor = intersection.color * d;
				color.x += lColor.x * dColor.x;
				color.y += lColor.y * dColor.y;
				color.z += lColor.z * dColor.z;

				// Calculate the specular lighting.
				// This is the reflect vector.
				float3 r = nLightVector - intersection.normal * 2.0f * d;
				float sd = dot(ray.dir, r);

				if (sd < 0.0f)
				{
					float specular = powf(sd, SPECULAR_INTENSITY) * attenuation;
					color += make_float4(1.0f) * specular;
				}
			}
		}
	}

	return color;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Calculates the color at the first intersection of the given ray. This takes the surface color, lighting and reflections into
// account.
//---------------------------------------------------------------------------------------------------------------------------------
__device__ float4 calculateRayColor(const Ray& ray)
{
	// Since recursion is not supported on the GPU, we use a stack. The stack size directly affects the amount of local memory used by
	// the kernel. 
	unsigned int depth = 0;
	float4 stackC[MAX_RECURSION];
	float stackR[MAX_RECURSION];
	Ray currentRay = ray;
	bool hit = true;
	Intersection intersection;

	do
	{
		// Calculate the closest intersection.
		hit = calculateClosestIntersection(currentRay, intersection, EPSILON, MAX_DISTANCE, _lights.Enable && depth == 0);

		if (hit)
		{
			// Calculate and store the color at the intersection point.
			stackC[depth] = calculateLighting(ray, intersection);
			// Store the reflection value at the intersection point.
			stackR[depth] = intersection.reflection;

			// If we hit a reflective surface and if the recursion is allowed to continue, construct the reflection ray that 
			// will be used for the next iteration of the loop.
			if (depth < MAX_RECURSION && intersection.reflection > EPSILON)
			{
				currentRay.origin = intersection.point;
				currentRay.dir = normalize(currentRay.dir - intersection.normal * 2.0f * dot(currentRay.dir, intersection.normal));
			}
		}
		else
		{
			// If there is no intersection, the ray has hit the void. Set the color to black and the reflection value to 0 (no reflection).
			stackC[depth] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			stackR[depth] = 0.0f;
		}

		++depth;
	}
	// Continue with the loop as long as we're allowed to and if we've hit a reflective surface.
	while (depth < MAX_RECURSION && hit && intersection.reflection > EPSILON);

	// Calculate the final color by walking the stack.
	float4 finalColor = stackC[depth - 1];
	for (int i = depth - 2; i >= 0; --i)
	{
		// The color is c(i) * (1 - r(i)) + c(i + 1) * r(i)
		float4 color = stackC[i] * (1.0f - stackR[i]);
		float4 reflectedColor = finalColor * stackR[i];

		finalColor = color + reflectedColor;
	}

	return finalColor;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Constructs a ray for the given pixel and writes the raytracing result to the render target.
//---------------------------------------------------------------------------------------------------------------------------------
__device__ void raytrace(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const int x, const int y,
	const float3& camPos, const float3& camRotMat1, const float3& camRotMat2, const float3& camRotMat3)
{
	// Construct the ray. The ray origin is the current camera position, the ray direction is determined by the given x and y values.
	// This automatically gives us a perspective projection.
	Ray ray;
	ray.origin = camPos;
	float3 dir = make_float3((float)x - rtWidth / 2, (float)y - rtHeight / 2, 0.7f * rtWidth);

	// Rotate the direction using the camera's rotation matrix.
	ray.dir.x = dot(camRotMat1, dir);
	ray.dir.y = dot(camRotMat2, dir);
	ray.dir.z = dot(camRotMat3, dir);
	ray.dir = normalize(ray.dir);

	// Write the result to the render target.
	renderTarget[y * rtWidth + x] = rgbToInt(calculateRayColor(ray));
}

//---------------------------------------------------------------------------------------------------------------------------------
// Calculates the thread's x and y coordinates and performs the raytracing for the pixel.
//---------------------------------------------------------------------------------------------------------------------------------
__global__ void raytraceGpu(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const Triangle* const triangles, const unsigned int numTriangles,
	const Sphere* const spheres, const unsigned int numSpheres,
	const Light* const lights, const unsigned int numLights, const bool showLights,
	const float3 camPos, const float3 camRotMat1, const float3 camRotMat2, const float3 camRotMat3)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Copy data to shared memory
	if (threadIdx.x == 0)
	{
		for (unsigned int i = 0; i < numTriangles; ++i)
			_triangles.List[i] = triangles[i];
		_triangles.Num = numTriangles;

		for (unsigned int i = 0; i < numSpheres; ++i)
			_spheres.List[i] = spheres[i];
		_spheres.Num = numSpheres;

		for (unsigned int i = 0; i < numLights; ++i)
			_lights.List[i] = lights[i];
		_lights.Num = numLights;
		_lights.Enable = showLights;
	}

	__syncthreads();

	raytrace(renderTarget, rtWidth, rtHeight,
		x, y, camPos, camRotMat1, camRotMat2, camRotMat3);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Invokes the raytracing algorithm either for the GPU or for the CPU.
//---------------------------------------------------------------------------------------------------------------------------------
void raytraceScene(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const Triangle* const triangles, const unsigned int numTriangles,
	const Sphere* const spheres, const unsigned int numSpheres,
	const Light* const lights, const unsigned int numLights, const bool showLights,
	const float3 camPos, const float3 camRotMat1, const float3 camRotMat2, const float3 camRotMat3)
{
	dim3 block(8, 8, 1);
	dim3 grid(rtWidth / block.x, rtHeight / block.y, 1);
	raytraceGpu << <grid, block >> > (renderTarget, rtWidth, rtHeight,
		triangles, numTriangles,
		spheres, numSpheres,
		lights, numLights, showLights,
		camPos, camRotMat1, camRotMat2, camRotMat3);
}