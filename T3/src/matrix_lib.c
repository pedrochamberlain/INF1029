// Nome: Lucas Angel Larios Prado - 2020723
// Nome: Pedro Chamberlain Matos - 1710883

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>

#include "matrix_lib.h"

// O valor padrão de threads do módulo é 1.
int NUM_THREADS = 1;

struct scalar_matrix_thread_args {
    float* m_array_start;
    int m_array_length;
    float scalar;
};

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

Função: scalar_matrix_mult
--------------------------
inicia o processo de cálculo do produto de um valor escalar 
em uma matriz em diversas threads diferentes.

scalar_value: valor escalar utilizada no cálculo. 
matrix: matriz a ser utilizada no cálculo.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    float *m_curr;
    int arrays_per_thread, m_array_length;
    struct scalar_matrix_thread_args args[NUM_THREADS];

    if (validate_matrix_contents(matrix) == 0) return 0;

    m_curr = matrix->rows;
    arrays_per_thread = matrix->height / NUM_THREADS;
    m_array_length = thread_rows * matrix->width;

    for (int i = 0; i < NUM_THREADS; i++, m_curr += m_array_length) {
        args[i].m_array_start = m_curr;
        args[i].m_array_length = m_array_length;
        args[i].scalar = scalar_value;
    }

    initialize_threads(args, scalar_matrix_mult_routine, sizeof(struct scalar_matrix_thread_args))
    return 1;
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

Função: matrix_matrix_mult
--------------------------
faz o cálculo do produto entre duas matrizes A e B, armazenando o resultado numa matriz C.

a: matriz A, a ser utilizada no cálculo.
b: matriz B, a ser utilizada no cálculo.
c: matriz C, resultado armazenado do cálculo entre as matrizes A e B.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
    float *a_curr, *a_end, *a_column_end, 
          *b_curr, *b_end, *b_row_start, 
          *c_curr;
    __m256 scalar_a_avx, matrix_b_avx, matrix_c_avx, result_avx;

    if (validate_matrix_operations(a, b, c) == 0) return 0;

    a_curr = a->rows;
    a_column_end = a->rows + (a->width - 1);
    a_end = a->rows + (a->height * a->width);
    
    b_curr = b->rows;
    b_row_start = b->rows;
    b_end = b->rows + (b->height * b->width);
    
    c_curr = c->rows;

    for (; a_curr != a_end; a_curr++) {
        scalar_a_avx = _mm256_set1_ps(*a_curr);

        for (b_curr = b_row_start; b_curr != b_row_start + b->width; b_curr += 8) {
            matrix_b_avx = _mm256_load_ps(b_curr);
            matrix_c_avx = _mm256_load_ps(c_curr);

            result_avx = _mm256_fmadd_ps(scalar_a_avx, matrix_b_avx, matrix_c_avx);
			_mm256_store_ps(c_curr, result_avx);
            c_curr += 8;
        }

        if (b_curr != b_end) {
            c_curr -= c->width;
        }

        if (a_curr != a_column_end) {
            b_row_start = b_curr;
        } else {
            b_row_start = b->rows;
            a_column_end += a->width;
        }
    }

    return 1;
}

/* 

Função: set_number_threads
--------------------------
atualiza a variável global NUM_THREADS, que define 
o número de threads que devem ser inicializadas.

*/

void set_number_threads(int num_threads) {
    if (num_threads <= 0) {
        printf("ERROR: Number of threads is invalid (<= 0).");
        return;
    }

    NUM_THREADS = num_threads;
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