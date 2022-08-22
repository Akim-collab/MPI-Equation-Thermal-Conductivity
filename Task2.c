#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define a 90000000.0
#define b 90000000.0
#define l 1.0
#define c 1.0 

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Incorrect number of parameters!\n");
		return 1;
	}
	
	int N = atoi(argv[1]);
	double T = atof(argv[2]);

	int rc;
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		printf("Error starting MPI program. Terminating!\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Status status;
	int NumTasks, Rank;
	double h = l / (double)N;
	double tau = 0.3 * h * h / (c * c);
	int Tn = (int)(T / tau);
	
	MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
	
	int sk, fk;
	sk = (N / NumTasks) * Rank;
	if (Rank != NumTasks - 1) 
		fk = sk + (N / NumTasks);
	else
		fk = N;

	double *u0, *u1;
        u0 = (double*)malloc(sizeof(double) * (fk - sk));
        u1 = (double*)malloc(sizeof(double) * (fk - sk));
	double StartTimeParallel = MPI_Wtime();
	for (int n = 0; n < Tn; n++) {
		double left = a;
		double right = b;
		if (Rank % 2 == 0) {
			if (Rank > 0) {
				MPI_Recv(&left, 1, MPI_DOUBLE, Rank - 1, Rank - 1, MPI_COMM_WORLD, &status);
				MPI_Ssend(u0, 1, MPI_DOUBLE, Rank - 1, Rank, MPI_COMM_WORLD);
			}
			if (Rank < NumTasks - 1) {
				MPI_Recv(&right, 1, MPI_DOUBLE, Rank + 1, Rank + 1, MPI_COMM_WORLD, &status);
				MPI_Ssend(u0 + (fk - sk - 1), 1, MPI_DOUBLE, Rank + 1, Rank, MPI_COMM_WORLD);
			}
		}
		else {
			if (Rank > 0) {
				MPI_Ssend(u0, 1, MPI_DOUBLE, Rank - 1, Rank, MPI_COMM_WORLD);
				MPI_Recv(&left, 1, MPI_DOUBLE, Rank - 1, Rank - 1, MPI_COMM_WORLD, &status);
			}
			if (Rank < NumTasks - 1) {
				MPI_Ssend(u0 + fk - sk - 1, 1, MPI_DOUBLE, Rank + 1, Rank, MPI_COMM_WORLD);
				MPI_Recv(&right, 1, MPI_DOUBLE, Rank + 1, Rank + 1, MPI_COMM_WORLD, &status);
			}
		}

		for (int m = 0; m < fk - sk; m++) {
			if (Rank == 0 && m == 0) {
				u1[m] = a;
			} else if (Rank + 1 == NumTasks && m + 1 == fk - sk) {
				u1[m] = b;
			} else {
			double lp, rp;
			if (m > 0)
				lp = u0[m - 1];
			else
				lp = left;
			if (m + 1 < fk - sk)
				rp = u0[m + 1];
			else
				rp = right;
			u1[m] = u0[m] + 0.3 * (lp - 2 * u0[m] + rp);
			}
		}
		double *t = u0;
		u0 = u1;
		u1 = t;
	}
	
	if (Rank > 0) {
		MPI_Send(u0, fk - sk, MPI_DOUBLE, 0, Rank, MPI_COMM_WORLD);
	} else {
		double *u = (double *)malloc(N * sizeof(double));
    		memcpy(u, u0, (fk - sk) * sizeof(double));
    		for (int from = 1; from < NumTasks; from++) {
      			int rsize;
			if (from + 1 == NumTasks)
				rsize = N - from * (N / NumTasks);
			else
				rsize = N / NumTasks;
      			MPI_Recv(u + from * (N / NumTasks), rsize, MPI_DOUBLE, from, from, MPI_COMM_WORLD, &status);
		}
		double TimeParallel = MPI_Wtime() - StartTimeParallel;
		printf("Number of processes = %d\nNumber of segment splits = %d\n", NumTasks, N);
		printf("Parallel algorithm Time = %lf\n", TimeParallel);
		printf("Result from parallel algorithm:\n");
		for (int i = 0; i < 10; i++)
			printf("%f %f\n", h * i, u[i]);
	
		printf("...............\n");
		double *u0s, *u1s;
		u0s = (double*)malloc(N * sizeof(double));
		u1s = (double*)malloc(N * sizeof(double));
		for (int n = 0; n < N; n++) {
			u0s[n] = 0.0;
			u1s[n] = 0.0;
		}
		double StartTimeSequential = MPI_Wtime();
		for (int n = 0; n < Tn; n++) {
			for (int m = 0; m < N; m++) {
				if (m == 0){
					u1s[m] = a;
				} else if (m == N - 1) {
					u1s[m] = b;
				} else {
					double lp = a;
                        		double rp = b;
                        		if (m > 0)
                                		lp = u0s[m - 1];
                        		if (m < N - 1)
                                		rp = u0s[m + 1];
                        		u1s[m] = u0s[m] + 0.3 * (lp - 2 * u0s[m] + rp);
				}
                	}
                	double *t = u0s;
                	u0s = u1s;
                	u1s = t;	
		}
		double TimeSequential = MPI_Wtime() - StartTimeSequential;
                      
		printf("Sequential algorithm Time = %lf\n", TimeSequential);
		double SpeedUp = TimeSequential / TimeParallel;
		printf("Result from sequential algorithm:\n");
                for (int i = 0; i < 10; i++)
                        printf("%f %f\n", h * i, u0s[i]);
		printf("...............\n");
		free(u1s);
		free(u);
		free(u0s);
		printf("SpeedUp = %lf\n", SpeedUp);
		
	}
 	free(u0);
	free(u1);
	MPI_Finalize();
	return 0;
}

