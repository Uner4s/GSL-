#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

int main(){
//	Se declaran dos matrices de 2x2
	gsl_matrix * a = gsl_matrix_alloc(3, 3);
	//gsl_matrix * b = gsl_matrix_alloc(2, 2);
//  Los espacios de memoria de las matrices se hacen 0	
	gsl_matrix_set_zero(a);
	//gsl_matrix_set_zero(b);
	
	int i, j;
//  se inicializan las dos matrices	
	//for(i=0;i<=1;i++){
		//for(j=0;j<=1;j++){
			gsl_matrix_set(a, 0, 0, 34.0);
			gsl_matrix_set(a, 0, 1, 5.0);
			gsl_matrix_set(a, 0, 2, 6.0);
			
			gsl_matrix_set(a, 1, 0, 4.0);
			gsl_matrix_set(a, 1, 1, 23.0);
			gsl_matrix_set(a, 1, 2, 6.0);
			
			gsl_matrix_set(a, 2, 0, 12.0);
			gsl_matrix_set(a, 2, 1, 8.0);
			gsl_matrix_set(a, 2, 2, 54.0);
			//gsl_matrix_set(b, i, j, 2.0);
		//}
	//}
//
//	gsl_matrix_add(a, b);
	gsl_linalg_cholesky_decomp(a);
	gsl_linalg_cholesky_invert(a);
	
	printf("Matriz a:\n");
	for(i=0;i<=2;i++){
		for(j=0;j<=2;j++){
			printf("%lf ", gsl_matrix_get(a, i, j));
		}
		printf("\n");
	}

/*	printf("Matriz b:\n");
	for(i=0;i<=1;i++){
		for(j=0;j<=1;j++){
			printf("%2.lf", gsl_matrix_get(b, i, j));
		}
		printf("\n");
	}	
	*/
	gsl_matrix_free(a);
	//gsl_matrix_free(b);
	
	return 0;
}