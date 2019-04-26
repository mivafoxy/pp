#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>  
#include <mpi.h> 
#include <math.h> 

#define PI 3.141592653589793

double f(double x) {
	return 1 / (1 + x * x);
}

// средних прямоугольников
double f1(int myrank, int nprocs, int n) {
	double sum = 0, h = 1.0 / n;

	for (int i = myrank + 1; i <= n; i += nprocs) {
		sum += f(h*(i - 0.5));
	}

	return 4 * h * sum;
}

// трапеций
double f2(int myrank, int nprocs, int n) {
	double sum = 0, h = 1.0 / n;

	for (int i = myrank + 1; i <= n - 1; i += nprocs) {
		sum += f(i * h);
	}

	if (myrank == 0) {
		sum += (f(0) + f(1)) / 2.0;
	}

	return 4 * h * sum;
}

// симпсона
double f3(int myrank, int nprocs, int n) {
	double sum = 0, h = 1.0 / n;

	for (int i = myrank + 1; i <= n; i += nprocs) {
		sum += f((i - 0.5) * h);
	}
	sum *= 2;

	for (int i = myrank + 1; i <= n - 1; i += nprocs) {
		sum += f(i * h);
	}

	if (myrank == 0) {
		sum += (f(0) + f(1)) / 2.0;
	}

	return 4 * h * sum / 3.0;
}

int main(int argc, char *argv[]) {

	double pi, sum;
	int myrank, nprocs;

	double t1, t2, allTime = 0;

	int N[] = { 10,50,100,500,1000,5000,10000 };

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	FILE *f = fopen("res.txt", "w");

	for (int fun = 0; fun < 3; fun++) {
		fprintf(f, "Значение n\t Значение pi\t Время вычисления(в сек.)\t Ошибка вычисления\n");

		for (int i = 0, n = N[i]; i < 7; i++, n = N[i]) {
			t1 = MPI_Wtime();
			switch (fun) {
			case 0:
				// прямоугольник
				sum = f1(myrank, nprocs, n);
				break;
			case 1:
				// трапеция
				sum = f2(myrank, nprocs, n);
				break;
			case 2:
				// симпсон
				sum = f3(myrank, nprocs, n);
				break;
			}

			t2 = MPI_Wtime() - t1;

			MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&t2, &allTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			if (myrank == 0) {
				fprintf(f, "%d\t %.18lg\t %lg\t %lg\n", n, pi, allTime/(1.0*nprocs), fabs(pi - PI));
			}
		}
	}

	fclose(f);

	MPI_Finalize();
	return 0;
}