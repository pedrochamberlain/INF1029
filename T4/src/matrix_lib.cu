// Nome: Lucas Angel Larios Prado - 2020723
// Nome: Pedro Chamberlain Matos - 1710883

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <cuda_runtime.h>

#include "matrix_lib.h"

#define NUM_THREADS_PER_BLOCK_LIMIT 1024
#define MAX_BLOCKS_PER_GRID_LIMIT 65535

// O valor padrão de threads por bloco é 256.
int NUM_THREADS_PER_BLOCK = 256;

// O valor limite padrão de blocos por grid é 4096.
int MAX_BLOCKS_PER_GRID = 4096;

/* 

Função: validate_matrix_contents
-------------------------------
valida se a instância de matriz é válida.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int validate_matrix_contents(struct matrix *matrix) {
    if (matrix == NULL) {
        printf("ERROR: Matrix is undeclared (NULL).");
        return 0;
    }

    if (matrix-> height < 0 || matrix->width < 0) {
        printf("ERROR: Matrix's height or width is invalid (< 0).");
        return 0;
    }

    return 1;
}

/* 

Função: validate_matrix_operations
----------------------------------
valida se as instâncias de matriz podem ser utilizadas para fazer um produto escalar e 
se a instância de matriz usada para armazenar o resultado é compátivel com as utilizadas no produto.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int validate_matrix_operations(struct matrix *a, struct matrix *b, struct matrix *c) {
    if (validate_matrix_contents(a) == 0  || validate_matrix_contents(b) == 0 || validate_matrix_contents(c) == 0) return 0;

    if (a->width != b->height) {
        printf("ERROR: Matrixes width and height don't match.");
        return 0;
    }

    if (a->height != c->height || b->width != c->width) {
        printf("ERROR: The resulting matrix's width and height don't match with the matrixes used in the scalar operation.");
        return 0;
    }

    return 1;
}

/* 

Função: set_grid_size
--------------------------
atualiza as variáveis globais NUM_THREADS_PER_BLOCK e
NUM_BLOCKS_PER_GRID, que definem o número de threads por
blocos e o número de blocos por grid que devem ser utilizados.

caso haja sucesso, a função retorna o valor 1.

caso algum dos parâmetros extrapole um dos valores máximos
definidos no início deste arquivo, os valores atuais devem 
ser mantidos e a função retorna o valor 0.

*/

void set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > NUM_THREADS_PER_BLOCK_LIMIT) {
        printf("ERROR: Number of threads per block exceeded value");
        return 0;
    } 
    
    if (max_blocks_per_grid > MAX_BLOCKS_PER_GRID_LIMIT) {
        printf("ERROR: Max number of blocks per grid exceeded value.");
        return 0;
    } 
    
    NUM_THREADS_PER_BLOCK = threads_per_block;
    MAX_BLOCKS_PER_GRID = max_blocks_per_grid;
    return 1;
}

/* 

Função: scalar_thread_routine
--------------------------
rotina iniciada por uma thread para fazer parte do processo 
de cálculo do produto de um valor escalar  em uma matriz.

*/

__global__
void scalar_thread_routine(int m_length, float *d_rows, float scalar_value) {
    unsigned long int 
        i = blockIdx.x * blockDim.x + threadIdx.x,
        stride = blockDim.x * gridDim.x;
    
    for (; i < m_length; i += stride)
        d_rows[i] *= scalar_value;
}

/* 

Função: scalar_matrix_mult
--------------------------
inicia o processo de cálculo do produto de um valor escalar em uma matriz.

scalar_value: valor escalar utilizada no cálculo. 
matrix: matriz a ser utilizada no cálculo.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    int m_length, num_blocks;

    if (validate_matrix_contents(matrix) == 0) return 0;
    
    m_length = matrix->m_width * matrix->m_height;
    num_blocks = (m_length + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    if (num_blocks > MAX_BLOCKS_PER_GRID) numBlocks = MAX_BLOCKS_PER_GRID;
    
    scalar_thread_routine<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(m_length, matrix->d_rows, scalar_value);
    cudaDeviceSynchronize();
    return 1;
}

/* 

Função: mult_thread_routine
--------------------------
rotina iniciada por uma thread para fazer parte do processo 
de cálculo do produto entre duas matrizes A e B, 
armazenando o resultado numa matriz C.

*/

int mult_thread_routine(int c_length, float *a_d_rows, float *b_d_rows, float *c_d_rows, int a_width, int b_width, int c_width) {
    unsigned long int 
        i = blockIdx.x * blockDim.x + threadIdx.x,
        stride = blockDim.x * gridDim.x;
        j, k, 
        a_line, a_end, b_index,
        c_line, c_column;

    for (; i < n; i += stride) {
        c_line = i / c_width;
        c_column = i % c_width;
        a_line = c_line * a_width;
        a_end = a_line + a_width;
        
        c_d_rows[i] = 0.0;
        
        for (j = a_line, k = 0; j < a_end; j++, k++) {
            b_index = k * b_width + c_column;
            c_d_rows[i] += a_d_rows[j] * b_d_rows[b_index];
        }
    }
}

/* 

Função: matrix_matrix_mult
--------------------------
inicia o processo do cálculo do produto entre duas matrizes A e B, 
armazenando o resultado numa matriz C.

a: matriz A, a ser utilizada no cálculo.
b: matriz B, a ser utilizada no cálculo.
c: matriz C, resultado armazenado do cálculo entre as matrizes A e B.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
    int c_length, num_blocks;
    
    if (validate_matrix_operations(a, b, c) == 0) return 0;

    c_length = c->height * c->width;
    num_blocks = (m_length + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    if (num_blocks > MAX_BLOCKS_PER_GRID) numBlocks = MAX_BLOCKS_PER_GRID;

    mult_thread_routine<<num_blocks, NUM_THREADS_PER_BLOCK>>(c_length, a->d_rows, b->d_rows, c->d_rows, a->width, b->width, c->width)

    cudaDeviceSynchronize();
    return 1;
}