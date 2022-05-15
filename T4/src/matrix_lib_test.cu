#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "matrix_lib.h"

extern "C" {
#include "timer.h"
}

#define HOST_DATASET_SIZE 4096000
#define DEVICE_DATASET_SIZE 1024000

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS_PER_GRID 4096

float scalar_value = 0.0f;

struct matrix matrixA, matrixB, matrixC;

int store_matrix(struct matrix *matrix, char *filename) {
  unsigned long int i = 0;
  unsigned long int N = 0;
  FILE *fd = NULL;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Try to open file of floats */
  if ((fd = fopen(filename, "wb")) == NULL) {
    printf("Unable to open file %s\n", filename);
    return 0;
  }

  float *nxt_a = matrix->h_rows;

  for ( i = 0;
        i < N;
        i += 8, nxt_a += 8) {

        if (fwrite(nxt_a, sizeof(float), 8, fd) != 8) {
            printf("Error writing to file %s: short write (less than 8 floats)\n", filename);
            return 0;
        }
  }

  if (fd != NULL) fclose(fd);

  return 1;
}

int load_matrix(struct matrix *matrix, char *filename) {
  unsigned long int i = 0;
  unsigned long int N = 0;
  FILE *fd = NULL;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Try to open file of floats */
  if ((fd = fopen(filename, "rb")) == NULL) {
    printf("Unable to open file %s\n", filename);
    return 0;
  }

  float *nxt_a = matrix->h_rows;

  for ( i = 0;
        i < N;
        i += 8, nxt_a += 8) {

        if (fread(nxt_a, sizeof(float), 8, fd) != 8) {
            printf("Error reading from file %s: short read (less than 8 floats)\n", filename);
            return 0;
        }
  }

  if (fd != NULL) fclose(fd);

  return 1;
}

int initialize_matrix(struct matrix *matrix, float value, float inc) {
  unsigned long int N;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  return 1;
}

int print_matrix(struct matrix *matrix) {
  unsigned long int i;
  unsigned long int N;
  unsigned long int nxt_newLine;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Initialize new line controol */
  nxt_newLine = matrix->width - 1;

  /* Print matrix elements */
  for (i = 0; i < N; i++) {
     printf("%5.1f ", matrix->h_rows[i]);
     if (i == nxt_newLine) {
        printf("\n");
        nxt_newLine += matrix->width;
     }
     if (i == 255) {
        printf("Ooops...256 printing limit found...skipping printing...\n");
        break;
     }
  }

  return 1;
}

int check_errors(struct matrix *matrix, float scalar_value) {
  unsigned long int i;
  unsigned long int N;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Check for errors (all values should be equal to scalar_value) */
  float maxError = 0.0f;
  float diffError = 0.0f;
  for (i = 0; i < N; i++)
    maxError = (maxError > (diffError=fabs(matrix->h_rows[i]-scalar_value)))? maxError : diffError;
  printf("Max error: %f\n", maxError);

  return 1;
}

int main(int argc, char **argv)
{
  unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
  unsigned long int threads_per_block, max_blocks_per_grid;
  char *matrixA_filename, *matrixB_filename, *result1_filename, *result2_filename;
  char *eptr = NULL;
  struct timeval start, stop, overall_t1, overall_t2;
  cudaError_t cudaError;

  // Mark overall start time
  gettimeofday(&overall_t1, NULL);

  // Disable buffering entirely
  setbuf(stdout, NULL);

  // Check arguments
  if (argc != 12) {
        printf("Usage: %s <scalar_value> <DimA_M> <DimA_N> <DimB_M> <DimB_N> <threads_per_block> <max_blocks_per_grid> <matrixA_filename> <matrixB_filename> <result1_filename> <result2_filename>\n", argv[0]);
        return 0;
  } else {
        //printf("Number of args: %d\n", argc);
        //for (int i=0; i<argc; ++i)
         //       printf("argv[%d] = %s\n", i, argv[i]);
  }

  // Convert arguments
  scalar_value = strtof(argv[1], NULL);
  DimA_M = strtol(argv[2], &eptr, 10);
  DimA_N = strtol(argv[3], &eptr, 10);
  DimB_M = strtol(argv[4], &eptr, 10);
  DimB_N = strtol(argv[5], &eptr, 10);
  threads_per_block = strtol(argv[6], &eptr, 10);
  max_blocks_per_grid = strtol(argv[7], &eptr, 10);
  matrixA_filename = argv[8];
  matrixB_filename = argv[9];
  result1_filename = argv[10];
  result2_filename = argv[11];

  if ((scalar_value == 0.0f) || (DimA_M == 0) || (DimA_N == 0) || (DimB_M == 0) || (DimB_N == 0)
      || (threads_per_block == 0) || (max_blocks_per_grid == 0)) {
        printf("%s: erro na conversao do argumento: errno = %d\n", argv[0], errno);

        /* If a conversion error occurred, display a message and exit */
        if (errno == EINVAL)
        {
            printf("Conversion error occurred: %d\n", errno);
            return 1;
        }

        /* If the value provided was out of range, display a warning message */
        if (errno == ERANGE) {
            printf("The value provided was out of rangei: %d\n", errno);
            return 1;
        }
  }

  /* Set GRID size */
  if (!set_grid_size(threads_per_block, max_blocks_per_grid)) {
        printf("%s: grid size setting problem.\n", argv[0]);
        return 1;
  }

  /* Allocate the arrays of the four matrixes ion the host */
  float *h_a =  (float*)aligned_alloc(32, DimA_M*DimA_N*sizeof(float));
  float *h_b = (float*)aligned_alloc(32, DimB_M*DimB_N*sizeof(float));
  float *h_c = (float*)aligned_alloc(32, DimA_M*DimB_N*sizeof(float));

  if ((h_a == NULL) || (h_b == NULL) || (h_c == NULL)) {
        printf("%s: array allocation problem.", argv[0]);
        return 1;
  }

  /* Allocate the arrays of the four matrixes ion the host */
  float *d_a, *d_b, *d_c;

  cudaError = cudaMalloc(&d_a, DimA_M*DimA_N*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
        printf("cudaMalloc d_a returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  cudaError = cudaMalloc(&d_b, DimB_M*DimB_N*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
        printf("cudaMalloc d_b returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  cudaError = cudaMalloc(&d_c, DimA_M*DimB_N*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
        printf("cudaMalloc d_c returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 0;
  }

  /* Initialize the three matrixes */
  matrixA.height = DimA_M;
  matrixA.width = DimA_N;
  matrixA.h_rows = h_a;
  matrixA.d_rows = d_a;
  //if (!initialize_matrix(&matrixA, 5.0f, 0.0f)) {
  if (!load_matrix(&matrixA, matrixA_filename)) {
        printf("%s: matrixA initialization problem.", argv[0]);
        return 1;
  }

  /* Print matrix */
  printf("---------- Matrix A ----------\n");
  print_matrix(&matrixA);

  matrixB.height = DimB_M;
  matrixB.width = DimB_N;
  matrixB.h_rows = h_b;
  matrixB.d_rows = d_b;
  //if (!initialize_matrix(&matrixB, 1.0f, 0.0f)) {
  if (!load_matrix(&matrixB, matrixB_filename)) {
        printf("%s: matrixB initialization problem.", argv[0]);
        return 1;
  }

  /* Print matrix */
  printf("---------- Matrix B ----------\n");
  print_matrix(&matrixB);

  matrixC.height = DimA_M;
  matrixC.width = DimB_N;
  matrixC.h_rows = h_c;
  matrixC.d_rows = d_c;

  /* Print matrix */
  printf("---------- Matrix C ----------\n");
  print_matrix(&matrixC);

  /* Scalar product of matrix A */
  printf("Executing scalar_matrix_mult(%5.1f, matrixA)...\n",scalar_value);
  gettimeofday(&start, NULL);
  if (!scalar_matrix_mult(scalar_value, &matrixA)) {
        printf("%s: scalar_matrix_mult problem.", argv[0]);
        return 1;
  }
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix A ----------\n");
  print_matrix(&matrixA);

  /* Write first result */
  printf("Writing first result: %s...\n", result1_filename);
  if (!store_matrix(&matrixA, result1_filename)) {
        printf("%s: failed to write first result to file.", argv[0]);
        return 1;
  }

  /* Calculate the product between matrix A and matrix B */
  printf("Executing matrix_matrix_mult(matrixA, mattrixB, matrixC)...\n");
  gettimeofday(&start, NULL);
  if (!matrix_matrix_mult(&matrixA, &matrixB, &matrixC)) {
        printf("%s: matrix_matrix_mult problem.", argv[0]);
        return 1;
  }
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Print matrix */
  printf("---------- Matrix C ----------\n");
  print_matrix(&matrixC);

  /* Write second result */
  printf("Writing second result: %s...\n", result2_filename);
  if (!store_matrix(&matrixC, result2_filename)) {
        printf("%s: failed to write second result to file.", argv[0]);
        return 1;
  }

  /* Check foor errors */
  printf("Checking matrixC for errors...\n");
  gettimeofday(&start, NULL);
  check_errors(&matrixC, 10240.0f);
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Free memory */
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  //Mark overall stop time
  gettimeofday(&overall_t2, NULL);
  // Show elapsed time
  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));
  // Return exit code
  return 0;
}
