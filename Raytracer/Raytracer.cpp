//---------------------------------------------------------------------------------------------------------------------------------
// Includes and usings
//---------------------------------------------------------------------------------------------------------------------------------
#include "glew.h"
#include "glfw3.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "Raytracer.cu.h"
#include <utMath.h>
#include <helper_math.h>

using namespace std;
using namespace Horde3D;

//---------------------------------------------------------------------------------------------------------------------------------
// Constants and macros
//---------------------------------------------------------------------------------------------------------------------------------
// The camera's movement speed.
#define MOVE_SPEED 300

// The window's width and height. This has almost no performance impact.
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

// The render target's width and height. This has a great performance impact.
#define RENDERTARGET_WIDTH 2560
#define RENDERTARGET_HEIGHT 1440

// Macro for safe cuda calls. If the call fails, the cuda error message is displayed and false is returned.
#define SAFE_CUDA(func) { (func); \
	cudaError_t error = cudaGetLastError(); \
	if (error != cudaSuccess) \
{ \
	cout << "Cuda error: " << cudaGetErrorString(error) << endl; \
	return false; \
} }

//---------------------------------------------------------------------------------------------------------------------------------
// Forward declarations 
//---------------------------------------------------------------------------------------------------------------------------------
void raytraceScene(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const Triangle* const triangles, const unsigned int numTriangles,
	const Sphere* const spheres, const unsigned int numSpheres,
	const Light* const lights, const unsigned int numLights, const bool showLights,
	const float3 camPos, const float3 camRotMat1, const float3 camRotMat2, const float3 camRotMat3);
bool initRts();
bool initOpenGL();
void presentScene();
bool raytrace();
int main();
bool initScene();
void addQuad(const float3& v1, const float3& v2, const float3& v3, const float3& v4, const float4& c, float reflection = 0.0f);
void initTriangle(const float3& v1, const float3& v2, const float3& v3, const float4& c, const float reflection = 0.0f);
void addLight(const float3& pos, const float4& color, const float radius);
void addSphere(const float3& pos, const float radius, const float4& color, const float reflection = 0.0f, bool lightSphere = false);
bool updateScene(GLFWwindow* window, float elapsedSeconds);
void transformTriangle(unsigned int idx, float x, float y, float z, float angleX, float angleY, float angleZ, float scale);
void transformLight(unsigned int idx, float x, float y, float z);
void transformSphere(unsigned int idx, float x, float y, float z);
void updateTexture(GLuint rt, GLuint rtTex);
void initRt(GLuint* rt, GLuint* rtTex);

//---------------------------------------------------------------------------------------------------------------------------------
// Global variables 
//---------------------------------------------------------------------------------------------------------------------------------
// The GPU render targets.
GLuint g_RtGpu;

// The render target textures.
GLuint g_RtGpuTex;

// The cuda view of the GPU render target.
cudaGraphicsResource* g_CudaRt;

// Indicates whether small spheres should be drawn at a light's position.
bool g_ShowLights = false;

// The current camera position.
float3 g_CamPos = make_float3(-350.0f, 250.0f, -500.0f);

// The current camera rotation.
Vec3f g_CamRot(degToRad(200), 0.0f, 0.0f);

// The first, second and third rows of the camera rotation matrix.
float3 g_CamRotMat1 = make_float3(0);
float3 g_CamRotMat2 = make_float3(0);
float3 g_CamRotMat3 = make_float3(0);

//---------------------------------------------------------------------------------------------------------------------------------
// Scene data
//
// The raytracer supports triangles, lights and spheres. For each of these the following data is stored:
// - The "untransformed" vector stores the original data
// - The "transformed" vector stores the transformed data
// - A pointer for both the CPU and the GPU version of the data
// - The number of objects.
//---------------------------------------------------------------------------------------------------------------------------------
vector<Triangle> g_UntransTriangles;
vector<Triangle> g_TransTriangles;
Triangle* g_GpuTriangles;
Triangle* g_CpuTriangles;
unsigned int g_NumTriangles;

vector<Light> g_UntransLights;
vector<Light> g_TransLights;
Light* g_GpuLights;
Light* g_CpuLights;
unsigned int g_NumLights;

vector<Sphere> g_UntransSpheres;
vector<Sphere> g_TransSpheres;
Sphere* g_GpuSpheres;
Sphere* g_CpuSpheres;
unsigned int g_NumSpheres;

//---------------------------------------------------------------------------------------------------------------------------------
// Initializes a render target and its texture.
//---------------------------------------------------------------------------------------------------------------------------------
void initRt(GLuint* rt, GLuint* rtTex)
{
	// Create the render target.
	glGenBuffers(1, rt);
	glBindBuffer(GL_ARRAY_BUFFER, *rt);

	// Allocate enough memory for the render target.
	GLuint size = RENDERTARGET_WIDTH * RENDERTARGET_HEIGHT * sizeof(unsigned int);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// Create the texture.
	glGenTextures(1, rtTex);
	glBindTexture(GL_TEXTURE_2D, *rtTex);

	// No magnification filtering.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Allocate enough memory for the texture.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RENDERTARGET_WIDTH, RENDERTARGET_HEIGHT, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Initializes the GPU and CPU render targets and registers the GPU render target with cuda. 
//---------------------------------------------------------------------------------------------------------------------------------
bool initRts()
{
	initRt(&g_RtGpu, &g_RtGpuTex);

	SAFE_CUDA(cudaGLSetGLDevice(0));
	SAFE_CUDA(cudaGraphicsGLRegisterBuffer(&g_CudaRt, g_RtGpu, cudaGraphicsMapFlagsWriteDiscard));

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Initializes OpenGL. Checks for pixel buffer object support.
//---------------------------------------------------------------------------------------------------------------------------------
bool initOpenGL()
{
	if (glewInit() != GLEW_OK)
	{
		cout << "GLEW initialization failed." << endl;
		return false;
	}

	if (!GLEW_ARB_pixel_buffer_object)
	{
		cout << "Pixel buffer objects not supported." << endl;
		return false;
	}

	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_BUFFER);
	glEnable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Draws the scene.
//---------------------------------------------------------------------------------------------------------------------------------
void presentScene()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, g_RtGpuTex);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 0.5f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 0.5f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, 0.5f);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, 1.0f, 0.5f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Copies the data from the render target to the render target texture.
//---------------------------------------------------------------------------------------------------------------------------------
void updateTexture(GLuint rt, GLuint rtTex)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, rt);
	glBindTexture(GL_TEXTURE_2D, rtTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RENDERTARGET_WIDTH, RENDERTARGET_HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Maps the render target, let's the CPU or GPU raytrace the scene and copies the updated render target data to the render
// target texture.
//---------------------------------------------------------------------------------------------------------------------------------
#include <assert.h>

bool raytrace()
{
	assert(g_NumTriangles < MAX_TRIANGLES);
	assert(g_NumSpheres < MAX_SPHERES);
	assert(g_NumLights < MAX_LIGHTS);

	SAFE_CUDA(cudaGraphicsMapResources(1, &g_CudaRt, 0));

	GLuint* rt;
	size_t rtSize;
	SAFE_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&rt, &rtSize, g_CudaRt));

	raytraceScene(rt, RENDERTARGET_WIDTH, RENDERTARGET_HEIGHT,
		g_GpuTriangles, g_NumTriangles,
		g_GpuSpheres, g_NumSpheres,
		g_GpuLights, g_NumLights, g_ShowLights,
		g_CamPos, g_CamRotMat1, g_CamRotMat2, g_CamRotMat3);

	SAFE_CUDA(cudaGraphicsUnmapResources(1, &g_CudaRt, 0));

	updateTexture(g_RtGpu, g_RtGpuTex);

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Iterates through all pixels and performs the raytracing for each pixel.
//---------------------------------------------------------------------------------------------------------------------------------
void raytrace(unsigned int* const renderTarget, const int rtWidth, const int rtHeight,
	const int x, const int y,
	const float3& camPos, const float3& camRotMat1, const float3& camRotMat2, const float3& camRotMat3);

//---------------------------------------------------------------------------------------------------------------------------------
// Resizes the OpenGL viewport.
//---------------------------------------------------------------------------------------------------------------------------------
void sizeChangedCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Toggles CPU and GPU mode. Toggles drawing the light spheres.
//---------------------------------------------------------------------------------------------------------------------------------
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == 'L' && action == GLFW_PRESS)
	{
		g_ShowLights = !g_ShowLights;
	}
}

//---------------------------------------------------------------------------------------------------------------------------------
// The entry point.
//---------------------------------------------------------------------------------------------------------------------------------
int main()
{
	// Initialize everything. In case of error, exit immediately.
	if (!glfwInit())
	{
		cout << "Window initialization failed." << endl;
		return 1;
	}

	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Raytracer", nullptr, nullptr);
	if (window == nullptr)
	{
		cout << "Window initialization failed." << endl;
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!initOpenGL())
	{
		cout << "OpenGL initialization failed." << endl;
		glfwTerminate();
		return 1;
	}

	if (!initRts())
	{
		cout << "Render Target initialization failed." << endl;
		glfwTerminate();
		return 1;
	}

	if (!initScene())
	{
		cout << "Scene initialization failed." << endl;
		glfwTerminate();
		return 1;
	}

	glfwSetKeyCallback(window, keyCallback);
	glfwSwapInterval(0);
	glfwSetWindowSizeCallback(window, sizeChangedCallback);

	bool running = true;
	double time = glfwGetTime();
	double timeProfile = glfwGetTime();

	glfwMaximizeWindow(window);
	glfwPollEvents();

	int width, height;
	glfwGetWindowSize(window, &width, &height);
	glfwSetCursorPos(window, width / 2, height / 2);

	while (!glfwWindowShouldClose(window))
	{
		double currentTime = glfwGetTime();
		double timeDelta = currentTime - time;
		time = currentTime;

		if (!updateScene(window, (float)timeDelta))
		{
			cout << "Scene update failed." << endl;
			break;
		}

		double currentTimeProfile = glfwGetTime();
		double timeDeltaProfile = currentTimeProfile - timeProfile;
		timeProfile = currentTimeProfile;

		if (!raytrace())
		{
			cout << "Raytracing failed." << endl;
			break;
		}

		ostringstream s;
		s << "Raytracer" << setprecision(2) << fixed << ", " << timeDeltaProfile * 1000 << "ms, " << (1 / timeDeltaProfile) << "fps";
		glfwSetWindowTitle(window, s.str().c_str());

		presentScene();
		glfwSwapBuffers(window);
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_ESCAPE))
		{
			break;
		}
	}

	cout << "Exiting..." << endl;

	cudaFree(g_GpuTriangles);
	glfwTerminate();

	return 0;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Initializes the scene.
//---------------------------------------------------------------------------------------------------------------------------------
bool initScene()
{
	// The cube.
	float4 color = make_float4(0.2f, 0.8f, 1.0f, 1.0f);
	float r = 0.3f;
	addQuad(make_float3(-1, -1, -1), make_float3(1, -1, -1), make_float3(1, 1, -1), make_float3(-1, 1, -1), color, r);
	addQuad(make_float3(-1, 1, -1), make_float3(1, 1, -1), make_float3(1, 1, 1), make_float3(-1, 1, 1), color, r);
	addQuad(make_float3(-1, -1, -1), make_float3(-1, -1, 1), make_float3(1, -1, 1), make_float3(1, -1, -1), color, r);
	addQuad(make_float3(-1, -1, 1), make_float3(-1, 1, 1), make_float3(1, 1, 1), make_float3(1, -1, 1), color, r);
	addQuad(make_float3(-1, -1, -1), make_float3(-1, 1, -1), make_float3(-1, 1, 1), make_float3(-1, -1, 1), color, r);
	addQuad(make_float3(1, -1, -1), make_float3(1, -1, 1), make_float3(1, 1, 1), make_float3(1, 1, -1), color, r);

	// The mirror.
	addQuad(make_float3(-1, 0, 0), make_float3(1, 0, 0), make_float3(1, 1, -0.2f), make_float3(-1, 1, -0.2f), make_float4(1, 1, 1, 1), 0.8f);
	addQuad(make_float3(-1, 0, 0), make_float3(-1, 1, -0.2f), make_float3(1, 1, -0.2f), make_float3(1, 0, 0), make_float4(1, 1, 1, 1));

	// The ground.
	float size = 3000;
	addQuad(make_float3(-size, 0, -size), make_float3(size, 0, -size), make_float3(size, 0, size), make_float3(-size, 0, size), make_float4(0.5f, 0.5f, 0.5f, 0.5f), 0.2f);

	// The lights.
	addLight(make_float3(-150, 300, 200), make_float4(1, 0.9f, 0.5f, 1), 1500);
	addLight(make_float3(100, 150, -300), make_float4(1, 0.2f, 0.2f, 1), 800);

	// The spheres.
	addSphere(make_float3(300, 200, -200), 150, make_float4(1, 1, 1, 1), 0.7f);
	addSphere(make_float3(-300, 120, 100), 100, make_float4(1, 1, 1, 1), 0.8f);

	// Register the triangles.
	SAFE_CUDA(cudaMalloc((void**)&g_GpuTriangles, sizeof(Triangle) * g_UntransTriangles.size()));
	SAFE_CUDA(cudaMemcpy(g_GpuTriangles, &g_TransTriangles[0], sizeof(Triangle) * g_UntransTriangles.size(), cudaMemcpyHostToDevice));
	g_CpuTriangles = &g_TransTriangles[0];
	g_NumTriangles = g_TransTriangles.size();

	// Register the lights.
	SAFE_CUDA(cudaMalloc((void**)&g_GpuLights, sizeof(Light) * g_UntransLights.size()));
	SAFE_CUDA(cudaMemcpy(g_GpuLights, &g_TransLights[0], sizeof(Light) * g_UntransLights.size(), cudaMemcpyHostToDevice));
	g_CpuLights = &g_TransLights[0];
	g_NumLights = g_TransLights.size();

	// Register the spheres.
	SAFE_CUDA(cudaMalloc((void**)&g_GpuSpheres, sizeof(Sphere) * g_UntransSpheres.size()));
	SAFE_CUDA(cudaMemcpy(g_GpuSpheres, &g_TransSpheres[0], sizeof(Sphere) * g_UntransSpheres.size(), cudaMemcpyHostToDevice));
	g_CpuSpheres = &g_TransSpheres[0];
	g_NumSpheres = g_TransSpheres.size();

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Adds a quad to the scene.
//---------------------------------------------------------------------------------------------------------------------------------
void addQuad(const float3& v1, const float3& v2, const float3& v3, const float3& v4, const float4& c, float reflection)
{
	initTriangle(v1, v2, v3, c, reflection);
	initTriangle(v1, v3, v4, c, reflection);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Adds a sphere to the scene.
//---------------------------------------------------------------------------------------------------------------------------------
void addSphere(const float3& pos, const float radius, const float4& color, const float reflection, bool lightSphere)
{
	Sphere s;
	s.pos = make_float4(pos, 0);
	s.color = color;
	s.params = make_float4(radius, reflection, (float)lightSphere, 0);
	g_UntransSpheres.push_back(s);
	g_TransSpheres.push_back(s);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Initializes a triangle and adds it to the scene.
//---------------------------------------------------------------------------------------------------------------------------------
void initTriangle(const float3& v1, const float3& v2, const float3& v3, const float4& c, const float reflection)
{
	Triangle t;
	t.vertex = make_float4(v1, 0);

	// Calculate the triangle's edges.
	float3 e1 = v2 - v1;
	float3 e2 = v3 - v1;
	t.edge1 = make_float4(e1, 0);
	t.edge2 = make_float4(e2, 0);
	t.color = c;
	// Calculate the normal.
	float3 normal = normalize(cross(e2, e1));
	t.normal = make_float4(normal, reflection);
	g_UntransTriangles.push_back(t);
	g_TransTriangles.push_back(t);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Adds a light to the scene.
//---------------------------------------------------------------------------------------------------------------------------------
void addLight(const float3& pos, const float4& color, const float radius)
{
	Light l;
	l.pos = make_float4(pos, radius);
	l.color = color;
	g_TransLights.push_back(l);
	g_UntransLights.push_back(l);

	// Add the light sphere to the scene.
	addSphere(pos, 5, color * 100, 0.0f, true);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Updates the scene and the camera.
//---------------------------------------------------------------------------------------------------------------------------------
bool updateScene(GLFWwindow* window, float elapsedSeconds)
{
	Vec3f move;
	double x, y;
	int width, height;

	if (glfwGetKey(window, 'W'))
	{
		move += Vec3f(0, 0, -1);
	}
	if (glfwGetKey(window, 'S'))
	{
		move += Vec3f(0, 0, 1);
	}
	if (glfwGetKey(window, 'A'))
	{
		move += Vec3f(1, 0, 0);
	}
	if (glfwGetKey(window, 'D'))
	{
		move += Vec3f(-1, 0, 0);
	}

	glfwGetCursorPos(window, &x, &y);
	glfwGetWindowSize(window, &width, &height);

	auto deltaX = x - width / 2;
	auto deltaY = y - height / 2;

	glfwSetCursorPos(window, width / 2, height / 2);

	// Use the input to update the camera.
	g_CamRot += Vec3f((float)deltaX, (float)deltaY, 0.0f) * 0.01f;
	g_CamRot.y = Horde3D::clamp(g_CamRot.y, -degToRad(89), degToRad(89));

	if (g_CamRot.x <= -Math::TwoPi || g_CamRot.x >= Math::TwoPi)
	{
		g_CamRot.x = 0;
	}

	Matrix4f rotMat = Matrix4f::RotMat(-g_CamRot.y, g_CamRot.x, 0);
	g_CamRotMat1 = make_float3(rotMat.c[0][0], rotMat.c[1][0], rotMat.c[2][0]);
	g_CamRotMat2 = make_float3(rotMat.c[0][1], rotMat.c[1][1], rotMat.c[2][1]);
	g_CamRotMat3 = make_float3(rotMat.c[0][2], rotMat.c[1][2], rotMat.c[2][2]);

	Vec3f rotatedMove = rotMat * move;
	Vec3f camPos = Vec3f(g_CamPos.x, g_CamPos.y, g_CamPos.z) + rotatedMove * (float)elapsedSeconds * MOVE_SPEED;
	g_CamPos = make_float3(camPos.x, camPos.y, camPos.z);

	// Update scene transformations.
	static float h;
	static float rm, rc;
	float totalSeconds = (float)glfwGetTime();
	h = 100 + sin(totalSeconds) * 50.0f;
	rc += elapsedSeconds / 5;
	rc = rc > Math::TwoPi ? 0 : rc;

	for (unsigned int i = 0; i < 12; ++i)
	{
		transformTriangle(i, 0.0f, h, 0.0f, 0.0f, rc, 0.0f, 50.0f);
	}

	for (unsigned int i = 12; i < 16; ++i)
	{
		transformTriangle(i, 0, 50, 400, 0, 0, 0, 600);
	}

	/*for (unsigned int i = 16; i < 20; ++i)
	{
		transformTriangle(i, 0, 50, -600, 0, 0, 0, 600);
	}*/

	static float3 lightPosition;
	lightPosition.x = sin(totalSeconds / 2) * 150;
	lightPosition.y = cos(totalSeconds) * 70;
	lightPosition.z = sin(totalSeconds * 2) * 350;

	transformLight(0, lightPosition.x, lightPosition.y, 0);
	transformLight(1, 0, 0, lightPosition.z);

	// Update the scene data on the device.
	SAFE_CUDA(cudaMemcpy(g_GpuTriangles, &g_TransTriangles[0], sizeof(Triangle) * g_TransTriangles.size(), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(g_GpuLights, &g_TransLights[0], sizeof(Light) * g_TransLights.size(), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(g_GpuSpheres, &g_TransSpheres[0], sizeof(Sphere) * g_TransSpheres.size(), cudaMemcpyHostToDevice));

	return true;
}

//---------------------------------------------------------------------------------------------------------------------------------
// Transforms a triangle.
//---------------------------------------------------------------------------------------------------------------------------------
void transformTriangle(unsigned int idx, float x, float y, float z, float angleX, float angleY, float angleZ, float scale)
{
	Matrix4f translate = Matrix4f::TransMat(x, y, z);
	Matrix4f rotate = Matrix4f::RotMat(angleX, angleY, angleZ);
	Matrix4f scaleMatrix = Matrix4f::ScaleMat(scale, scale, scale);

	Matrix4f mt = rotate * translate * scaleMatrix;
	Matrix4f mn = rotate * scaleMatrix;

	Triangle& t = g_TransTriangles[idx];
	Triangle& to = g_UntransTriangles[idx];

	// Recalculate the vertex, edges and normal.
	Vec3f v = mt * Vec3f(to.vertex.x, to.vertex.y, to.vertex.z);
	Vec3f e1 = mn * Vec3f(to.edge1.x, to.edge1.y, to.edge1.z);
	Vec3f e2 = mn * Vec3f(to.edge2.x, to.edge2.y, to.edge2.z);

	// Construct the normal matrix to transform the normal.
	Vec3f n = rotate.inverted().transposed() * Vec3f(to.normal.x, to.normal.y, to.normal.z);
	n.normalize();

	t.vertex = make_float4(v.x, v.y, v.z, 0);
	t.edge1 = make_float4(e1.x, e1.y, e1.z, 0);
	t.edge2 = make_float4(e2.x, e2.y, e2.z, 0);
	t.normal = make_float4(n.x, n.y, n.z, to.normal.w);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Transforms a light.
//---------------------------------------------------------------------------------------------------------------------------------
void transformLight(unsigned int idx, float x, float y, float z)
{
	Matrix4f m = Matrix4f::TransMat(x, y, z);

	Light& l = g_TransLights[idx];
	Light& lo = g_UntransLights[idx];

	Vec3f pos = m * Vec3f(lo.pos.x, lo.pos.y, lo.pos.z);
	l.pos = make_float4(pos.x, pos.y, pos.z, lo.pos.w);

	transformSphere(idx, x, y, z);
}

//---------------------------------------------------------------------------------------------------------------------------------
// Transforms a sphere.
//---------------------------------------------------------------------------------------------------------------------------------
void transformSphere(unsigned int idx, float x, float y, float z)
{
	Matrix4f m = Matrix4f::TransMat(x, y, z);

	Sphere& s = g_TransSpheres[idx];
	Sphere& so = g_UntransSpheres[idx];

	Vec3f pos = m * Vec3f(so.pos.x, so.pos.y, so.pos.z);
	s.pos = make_float4(pos.x, pos.y, pos.z, 0);
}