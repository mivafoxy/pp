#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <ctime>
# include <string.h>
#include <chrono>
#include "./common/inc/helper_image.h"

unsigned int width = 512, height = 512;
 // Загрузка изображения. Вырвана из лабы по cuda прошлого семестра.
void loadImage(char *file, unsigned char** pixels, unsigned int * width, unsigned int * height)
{
	size_t file_length = strlen(file);

	if (!strcmp(&file[file_length - 3], "pgm"))
	{
		if (sdkLoadPGM<unsigned char>(file, pixels, width, height) != true)
		{
			printf("Failed to load PGM image file: %s\n", file);
			exit(EXIT_FAILURE);
		}
	}
	return;
}


// Сохранение изображения. Также вырвано из лабы по cuda прошлого семестра.
void saveImage(char *file, unsigned char* pixels, unsigned int width, unsigned int  height)
{
	size_t file_length = strlen(file);
	if (!strcmp(&file[file_length - 3], "pgm"))
	{
		sdkSavePGM(file, pixels, width, height);
	}
	return;
}

// Box blur. В каждом потоке обрабатывается часть картинки и выставляется новое значение пикселю, которое вычисляется по формуле с коэффициентом размытия (радиус). 
// Чем больше этот коэффициент, тем больше размытие. Под размытием подразумевается усреднение цвета пикселя таким образом, чтобы он переставал "контрастировать" с соседними.
const char *g_pcszSource =
"__kernel void func(__global unsigned char * im_s,__global unsigned char * im_d,unsigned int w,unsigned int h,int radius) \n"
"{\n"
/*Идентификатор процесса.*/"	int i = get_global_id(0);\n"
/*Координата по х на картинке.*/	"	int tidx = i / w;\n"
/*Координата по у на картинке.*/"	int tidy = i % w;\n"
/*Если внутри картинки (в её пределах)*/"	if (tidx < h && tidy < w)\n"
"	{\n"
/*Радиус для блюра.*/"		unsigned int r = 0;\n"
/*Идём по строкам и по колонкам пикселей картинки.*/"		for (int ir = -radius; ir <= radius; ir++)\n"
"			for (int ic = -radius; ic <= radius; ic++)\n"
"			{\n"
/*Временные переменные для swap - а пикселей.*/"				int temp_x, temp_y;\n"
"				if (tidx + ir < 0) {\n"
"					temp_x = -(tidx + ir);\n"
"				}\n"
"				else if (tidx + ir >= h) {\n"
"					temp_x = (h-1) - ((tidx + ir) - h);\n"
"				}\n"
"				else {\n"
"					temp_x = tidx + ir;\n"
"				}\n"
"				if (tidy + ic < 0) {\n"
"					temp_y = -(tidy + ic);\n"
"				}\n"
"				else if (tidy + ic >= w) {\n"
"					temp_y = (w-1) - ((tidy + ic) - w);\n"
"				}\n"
"				else {\n"
"					temp_y = tidy + ic;\n"
"				}\n"
"				r += im_s[temp_x*w + temp_y];\n"
"			}\n"
/*Формула для перерасчёта радиуса. */"		r /= ((2 * radius + 1)*(2 * radius + 1));\n"
/*Результат обработки box blur-ом*/"		im_d[tidx*w + tidy] = (unsigned char)r;\n"
"	}\n"
"}\n";
int main(int argc, char ** argv)
{
	unsigned char * im_s = NULL; // Оригинальное изображение.
	unsigned char * d_pixels = NULL; // Результирующее изображение.

	int radius = 5;
	int N = 10;

	char * src_path = (char*)"lena.pgm";
	char * d_result_path = (char*)"lena_box_blur.pgm";

	loadImage(src_path, &im_s, &width, &height); // Загрузка изображения с записью ее высоты и ширины.

	printf("Image size %dx%d\n", width, height);
	int image_size = sizeof(unsigned char) * width * height;

	// Код, взятый с предыдущего примера.

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


	
	for (int i = radius; i < radius + N; i++) {

		// инициализация
		cl_mem buffer_im_s = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size, im_s, NULL);
		cl_mem buffer_im_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_size, im_s, NULL);

		// установка аргументов
		clSetKernelArg(kernel, 0, sizeof(buffer_im_s), (void *)&buffer_im_s);
		clSetKernelArg(kernel, 1, sizeof(buffer_im_d), (void *)&buffer_im_d);
		clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&width);
		clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&height);
		clSetKernelArg(kernel, 4, sizeof(int), (void *)&i);

		// запуск ядра
		cl_event event;
		size_t uGlobalWorkSize = width * height;

		clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &uGlobalWorkSize, NULL, 0, NULL, &event);

		clWaitForEvents(1, &event);
		clFinish(queue);

		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		// отображение результата
		d_pixels = new unsigned char[width*height];
		cl_uchar *puData = (cl_uchar *)clEnqueueMapBuffer(queue, buffer_im_d, CL_TRUE, CL_MAP_READ, 0, image_size, 0, NULL, NULL, NULL);
		for (int i = 0; i < width*height; ++i) {
			//std::cout << i << " = " << puData[i] << "; ";
			d_pixels[i] = puData[i];
		}
		char* fileName = new char[17];
		sprintf(fileName, "result_bb_%02d.pgm", i);
		saveImage(fileName, d_pixels, width, height);
		//std::cout << std::endl;
		clEnqueueUnmapMemObject(queue, buffer_im_d, puData, 0, NULL, NULL);
		//

		clReleaseMemObject(buffer_im_s);
		clReleaseMemObject(buffer_im_d);
		double t1 = (time_end - time_start) / 1000000000.0;


		res << i << "\t"<< t1 << std::endl;
	}


	res.close();

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	delete[] pPlatforms;
	delete d_pixels;

	system("pause");
}
