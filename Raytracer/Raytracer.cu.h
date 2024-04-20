#pragma once

//---------------------------------------------------------------------------------------------------------------------------------
// Includes and usings
//---------------------------------------------------------------------------------------------------------------------------------
#include <vector_types.h>
#include <vector_functions.h>

//---------------------------------------------------------------------------------------------------------------------------------
// Constants and macros
//---------------------------------------------------------------------------------------------------------------------------------
// Very small epsilon for checking against 0.0f
#define EPSILON 0.001f

// The maximum distance from the ray origin to an intersection.
#define MAX_DISTANCE 9999999.9f 

// The maximum recursion depth when following reflection rays.
#define MAX_RECURSION 6

// The intensity of the ambient color.
#define AMBIENT_INTENSITY 0.1f

// The intensity of the specular light.
#define SPECULAR_INTENSITY 32.0f

//---------------------------------------------------------------------------------------------------------------------------------
// Structures
//---------------------------------------------------------------------------------------------------------------------------------
// A ray.
struct Ray
{
	float3 origin;
	float3 dir;
};

// A triangle. We store one vertex and the two edges. This avoids recalculating the edges during every intersection test.
struct Triangle
{
	float4 vertex;
	float4 edge1;
	float4 edge2;
	float4 color;
	float4 normal; // normal.w is the reflection value
};

// A point light.
struct Light
{
	float4 pos; // pos.w is the light radius
	float4 color;
};

// A sphere.
struct Sphere
{
	float4 pos;
	float4 color;
	// params.x: radius
	// params.y: reflection value
	// params.z: 1 -> light sphere
	float4 params;
};

// An intersection result.
struct Intersection
{
	// The location of the intersection.
	float3 point;
	// The reflection value.
	float reflection;
	// The color at the intersection point.
	float4 color;
	// The normal at the intersection.
	float3 normal;
};

#define MAX_TRIANGLES 32
#define MAX_SPHERES 8
#define MAX_LIGHTS 4

struct TriangleList
{
	Triangle List[MAX_TRIANGLES];
	unsigned int Num;
};

struct SphereList
{
	Sphere List[MAX_SPHERES];
	unsigned int Num;
};

struct LightList
{
	Light List[MAX_LIGHTS];
	unsigned int Num;
	bool Enable;
};