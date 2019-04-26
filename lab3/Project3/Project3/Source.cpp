#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

double a = 3.3;
double b = 4.4;

double* init(int n) {
	double* ar = new double[n];
	for (int i = 0; i < n; i++) {
		ar[i] = rand() % 100 / 10.0;
	}
	return ar;
}

const char *g_pcszSource =
"__kernel void func(double a,__global const double * x,double b,__global const double * y,__global const double * z,__global double * w,int n) \n"
"{ \n"
"int i = get_global_id(0); \n"
"if (i>=n) return; \n"
"w[i] = a * x[i] + b * y[i] * z[i]; \n"
"} \n";

int main()
{

	cl_uint uNumPlatforms;
	clGetPlatformIDs(0, NULL, &uNumPlatforms);
	cl_platform_id *pPlatforms = new cl_platform_id[uNumPlatforms];
	clGetPlatformIDs(uNumPlatforms, pPlatforms, &uNumPlatforms);

	cl_device_id deviceID;
	cl_uint uNumGPU;
	clGetDeviceIDs(pPlatforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &uNumGPU);

	cl_int errcode_ret;
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &errcode_ret);

	errcode_ret = 0;
	cl_command_queue queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);

	errcode_ret = CL_SUCCESS;
	size_t source_size = strlen(g_pcszSource);
	cl_program program = clCreateProgramWithSource(context, 1, &g_pcszSource, (const size_t *)&source_size, &errcode_ret);

	cl_int errcode = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "func", NULL);

	std::ofstream res("text.txt");
	res << "Время (с.)\tВремя CPU (с.)" << std::endl;

	int N = 256;
	//while (N < 10000000) {
		std::cout << N << std::endl;
		double* x = init(N), *y = init(N), *z = init(N), *w = new double[N];

		// инициализация
		cl_mem buffer_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), x, NULL);
		cl_mem buffer_y = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), y, NULL);
		cl_mem buffer_z = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), z, NULL);
		cl_mem buffer_w = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(cl_double), NULL, NULL);

		// установка аргументов
		clSetKernelArg(kernel, 0, sizeof(double), (void *)&a);
		clSetKernelArg(kernel, 1, sizeof(buffer_x), (void *)&buffer_x);
		clSetKernelArg(kernel, 2, sizeof(double), (void *)&b);
		clSetKernelArg(kernel, 3, sizeof(buffer_y), (void *)&buffer_y);
		clSetKernelArg(kernel, 4, sizeof(buffer_z), (void *)&buffer_z);
		clSetKernelArg(kernel, 5, sizeof(buffer_w), (void *)&buffer_w);
		clSetKernelArg(kernel, 6, sizeof(N), (void *)&N);

		// запуск ядра
		cl_event event;
		size_t uGlobalWorkSize = N;

		clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &uGlobalWorkSize, NULL, 0, NULL, &event);

		clWaitForEvents(1, &event);
		clFinish(queue);

		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		// отображение результата
		cl_double *puData = (cl_double *)clEnqueueMapBuffer(queue, buffer_w, CL_TRUE, CL_MAP_READ, 0, N * sizeof(cl_double), 0, NULL, NULL, NULL);
		for (int i = 0; i < N; ++i) {
			//std::cout << i << " = " << puData[i] << "; ";
			w[i] = puData[i];
		}
		//std::cout << std::endl;
		clEnqueueUnmapMemObject(queue, buffer_w, puData, 0, NULL, NULL);
		//

		auto time2_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < N; i++) {
			w[i] = a * x[i] + b * y[i] * z[i];
		}
		auto time2_end = std::chrono::high_resolution_clock::now();
		//std::cout << (double)(time2) / CLOCKS_PER_SEC << std::endl;

		clReleaseMemObject(buffer_x);
		clReleaseMemObject(buffer_y);
		clReleaseMemObject(buffer_z);
		clReleaseMemObject(buffer_w);
		delete x, y, z, w;

		N *= 2;

		double t1 = (time_end - time_start) / 1000000000.0;
		double t2 = (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(time2_end - time2_start).count()) / 1e9;

		res << t1  << "\t" << t2 <<"\t"<< t2/(double)t1 << std::endl;
	//}

	res.close();

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	delete[] pPlatforms;

	system("pause");
}
