#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include "matrix_lib.h"

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
faz o cálculo do produto de um valor escalar em uma matriz.

scalar_value: valor escalar utilizada no cálculo. 
matrix: matriz a ser utilizada no cálculo.

retorna: caso haja sucesso, a função retorna o valor 1. em caso de erro, a função deve retornar 0.

*/

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    float *m_curr, *m_end;
    __m256 m_curr_reg, scalar_value_reg, result_reg;
    if (validate_matrix_contents(matrix) == 0) return 0;
    
    m_curr = matrix->rows;
    m_end = matrix->rows + (matrix->height * matrix->width);
    scalar_value_reg = _mm256_set1_ps(scalar_value);
    
    for (; m_curr <= m_end; m_curr++) {
        m_curr_reg = _mm256_load_ps(m_curr);
		result_reg = _mm256_mul_ps(scalar_value_reg, m_curr);
		_mm256_store_ps(m_curr, result_reg);
    }

    return 1;
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

    if (validate_matrix_operations(a, b, c) == 0) return 0;

    a_curr = a->rows;
    a_column_end = a->rows + (a->width - 1);
    a_end = a->rows + (a->height * a->width);
    
    b_curr = b->rows;
    b_row_start = b->rows;
    b_end = b->rows + (b->height * b->width);
    
    c_curr = c->rows;

    for (; a_curr != a_end; a_curr++) {
        for (b_curr = b_row_start; b_curr != b_row_start + b->width; b_curr++) {
            if (*c_curr == 0.0f) {
                *c_curr = *a_curr * (*b_curr);
            } else {
                *c_curr += *a_curr * (*b_curr);
            }

            c_curr++;
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