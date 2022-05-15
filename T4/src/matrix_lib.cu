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

Função: initialize_threads
--------------------------
inicializa as threads que serão utilizadas para efetuar 
os cálculos das funções scalar_matrix_mult e matrix_matrix_mult.

*/

void initialize_threads(void *thread_routine, void *args, int args_struct_size) {
    pthread_t threads[NUM_THREADS]; 
    pthread_attr_t thread_attr;
    void *value_ptr;

    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    for(int i = 0; i < NUM_THREADS; i++, args += args_struct_size) {
        pthread_create(&threads[i], &thread_attr, thread_routine, args);
        pthread_join(threads[i], &value_ptr);
    }
}

/* 

Função: scalar_matrix_mult_routine
--------------------------
rotina iniciada por uma thread para fazer parte do processo 
de cálculo do produto de um valor escalar  em uma matriz.

thread_args: parâmetros da thread. 

para mais informações sobre esses parâmetros, verifique 
a definição da struct scalar_matrix_thread_args.

*/

int scalar_matrix_mult_routine(void *thread_args) {
    struct scalar_matrix_thread_args *args = 
        (struct scalar_matrix_thread_args *) thread_args;

    float *m_curr = args->m_array_start, 
        *m_end = args->m_array_start + args->m_array_length;

    __m256 curr, result, 
        scalar = _mm256_set1_ps(args->scalar);

    for (; m_curr <= m_end; m_curr += 8) {
        curr = _mm256_load_ps(m_curr);
        result = _mm256_mul_ps(curr, scalar);
        _mm256_store_ps(m_curr, result);
    }

    pthread_exit(NULL);
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
    float *m_curr;
    int rows_per_thread, m_array_length;
    struct scalar_matrix_thread_args args[NUM_THREADS];

    if (validate_matrix_contents(matrix) == 0) return 0;

    m_curr = matrix->rows;
    rows_per_thread = matrix->height / NUM_THREADS;
    m_array_length = rows_per_thread * matrix->width;

    for (int i = 0; i < NUM_THREADS; i++, m_curr += m_array_length) {
        args[i].m_array_start = m_curr;
        args[i].m_array_length = m_array_length;
        args[i].scalar = scalar_value;
    }

    initialize_threads(scalar_matrix_mult_routine, args, sizeof(struct scalar_matrix_thread_args));
    return 1;
}

/* 

Função: matrix_matrix_mult_routine
--------------------------
rotina iniciada por uma thread para fazer parte do processo 
de cálculo do produto entre duas matrizes A e B, 
armazenando o resultado numa matriz C.

thread_args: parâmetros da thread. 

para mais informações sobre esses parâmetros, verifique 
a definição da struct matrix_matrix_thread_args.

*/

int matrix_matrix_mult_routine(void *thread_args) {
    struct matrix_matrix_thread_args *args = 
        (struct matrix_matrix_thread_args *) thread_args;

    int a_column = 0, 
        a_row = 0;

    float *a_curr = args->a_start,
        *b_curr, *c_curr;

    __m256 matrix_a_avx, matrix_b_avx, matrix_c_avx, result_avx;

    for (; a_row < args->rows_per_thread; a_curr++) {
        matrix_a_avx = _mm256_set1_ps(*a_curr);

        b_curr = args->b_start;
        b_curr += args->b_width * a_column;

        c_curr = args->c_start;
        c_curr += args->c_width * a_row;

        for (int curr_column = 0; curr_column < args->b_width; curr_column += 8, b_curr += 8, c_curr += 8) {
            matrix_b_avx = _mm256_load_ps(b_curr);
            matrix_c_avx = _mm256_load_ps(c_curr);
            result_avx = _mm256_fmadd_ps(matrix_a_avx, matrix_b_avx, matrix_c_avx);
			_mm256_store_ps(c_curr, result_avx);
        }

        if (a_column + 1 == args->a_width) {
            a_column = 0;
            a_row++;
        } else {
            a_column++;
        }
    }

    pthread_exit(NULL);
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
    float *a_curr, *c_curr;
    int rows_per_thread, a_array_length, c_array_length;
    struct matrix_matrix_thread_args args[NUM_THREADS];

    if (validate_matrix_operations(a, b, c) == 0) return 0;

    a_curr = a->rows;
    c_curr = c->rows;
    rows_per_thread = c->height / NUM_THREADS;
    a_array_length = rows_per_thread * a->width;
    c_array_length = rows_per_thread * c->width;

    for (int i = 0; i < NUM_THREADS; i++, a_curr += a_array_length, c_curr += c_array_length) {
        args[i].a_start = a_curr;
        args[i].b_start = b->rows;
        args[i].c_start = c_curr;
        args[i].a_width = a->width;
        args[i].b_width = b->width;
        args[i].c_width = c->width;
        args[i].rows_per_thread = rows_per_thread;
    }

    initialize_threads(matrix_matrix_mult_routine, args, sizeof(struct matrix_matrix_thread_args));
    return 1;
}