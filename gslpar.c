#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h> 
#define size 50000

double matriz_X[size][500];
double matriz_Y[size][1];
double matriz_X_T[500][size];
double matriz_XTX[500][size];
double matriz_INV[500][500];
double matriz_AUX[size][500];

double timeval_diff(struct timeval *a, struct timeval *b)
{
  return
  	(double)(a->tv_sec + (double)a->tv_usec/1000000) -
    (double)(b->tv_sec + (double)b->tv_usec/1000000);
}




int filas = 47000;
int columnas = 301;

int intervalo1;
int intervalo2;
int intervalo3;

//***************************************************************************************************************

void *threadOne(){
    int i,j;
    int t,h;
	gsl_matrix * matrizXT = gsl_matrix_alloc(columnas, filas);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix * matrizXTX = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix_set_zero(matrizXTX);
	gsl_matrix_set_zero(matrizXT);
	gsl_matrix_set_zero(matrizX);

    for(i=0;i<columnas;i++){
		for(j=0;j<filas;j++){
			matriz_X_T[i][j] = matriz_X[j][i];
			gsl_matrix_set(matrizXT, i, j, matriz_X[j][i]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<filas;i++) {
		for(j=0;j<intervalo1;j++){
			gsl_matrix_set(matrizX, h, t, matriz_X[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXT, matrizX, 0.0, matrizXTX); 


    h = 0;
    t = 0;
	for(i=0;i<columnas;i++) {
		for(j=0;j<intervalo1;j++){
			matriz_XTX[i][j] = gsl_matrix_get(matrizXTX,i,j) ;
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizXT);
	gsl_matrix_free(matrizXTX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadTwo(){
    int i,j;
    int t,h;
	gsl_matrix * matrizXT = gsl_matrix_alloc(columnas, filas);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix * matrizXTX = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix_set_zero(matrizXTX);
	gsl_matrix_set_zero(matrizXT);
	gsl_matrix_set_zero(matrizX);

    for(i=0;i<columnas;i++){
		for(j=0;j<filas;j++){
			matriz_X_T[i][j] = matriz_X[j][i];
			gsl_matrix_set(matrizXT, i, j, matriz_X[j][i]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<filas;i++) {
		for(j=intervalo1;j<intervalo2;j++){
			gsl_matrix_set(matrizX, h, t, matriz_X[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXT, matrizX, 0.0, matrizXTX); 
	

    h = 0;
    t = 0;
	for(i=0;i<columnas;i++) {
		for(j=intervalo1;j<intervalo2;j++){
			matriz_XTX[i][j] = gsl_matrix_get(matrizXTX,h,t);
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizXT);
	gsl_matrix_free(matrizXTX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadThree(){
    int i,j;
    int t,h;
	gsl_matrix * matrizXT = gsl_matrix_alloc(columnas, filas);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix * matrizXTX = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix_set_zero(matrizXTX);
	gsl_matrix_set_zero(matrizXT);
	gsl_matrix_set_zero(matrizX);

    for(i=0;i<columnas;i++){
		for(j=0;j<filas;j++){
			matriz_X_T[i][j] = matriz_X[j][i];
			gsl_matrix_set(matrizXT, i, j, matriz_X[j][i]);
		}
    }

    h = 0;
    t = 0;   
    
    for(i=0;i<filas;i++) {
		for(j=intervalo2;j<intervalo3;j++){
			gsl_matrix_set(matrizX, h, t, matriz_X[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXT, matrizX, 0.0, matrizXTX); 
	

	h = 0;
    t = 0;
	for(i=0;i<columnas;i++) {
		for(j=intervalo2;j<intervalo3;j++){
			matriz_XTX[i][j] = gsl_matrix_get(matrizXTX,h,t);
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizXT);
	gsl_matrix_free(matrizXTX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadFour(){
    int i,j;	
    int t,h;
	gsl_matrix * matrizXT = gsl_matrix_alloc(columnas, filas);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas - intervalo3);
	gsl_matrix * matrizXTX = gsl_matrix_alloc(columnas, columnas - intervalo3);
	gsl_matrix_set_zero(matrizXTX);
	gsl_matrix_set_zero(matrizXT);
	gsl_matrix_set_zero(matrizX);

    for(i=0;i<columnas;i++){
		for(j=0;j<filas;j++){
			matriz_X_T[i][j] = matriz_X[j][i];
			gsl_matrix_set(matrizXT, i, j, matriz_X[j][i]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<filas;i++) {
		for(j=intervalo3;j<columnas;j++){
			gsl_matrix_set(matrizX, h, t, matriz_X[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXT, matrizX, 0.0, matrizXTX); 
	
    h = 0;
    t = 0;
	for(i=0;i<columnas;i++) {
		for(j=intervalo3;j<columnas;j++){
			matriz_XTX[i][j] = gsl_matrix_get(matrizXTX,h,t) ;
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizXT);
	gsl_matrix_free(matrizXTX);
	pthread_exit(0);
}

//***************************************************************************************************************




//***************************************************************************************************************

void *threadFive(){
    int i,j;
    int t,h;
	gsl_matrix * matrizINV = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizAUX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix_set_zero(matrizINV);
	gsl_matrix_set_zero(matrizX);
	gsl_matrix_set_zero(matrizAUX);

    for(i=0;i<filas;i++){
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizX, i, j, matriz_X[i][j]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<columnas;i++) {
		for(j=0;j<intervalo1;j++){
			gsl_matrix_set(matrizINV, h, t, matriz_INV[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizX, matrizINV, 0.0, matrizAUX); 

    h = 0;
    t = 0;
	for(i=0;i<filas;i++) {
		for(j=0;j<intervalo1;j++){
			matriz_AUX[i][j] = gsl_matrix_get(matrizAUX,i,j);
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizINV);
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizAUX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadSix(){
    int i,j;
    int t,h;
	gsl_matrix * matrizINV = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizAUX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix_set_zero(matrizINV);
	gsl_matrix_set_zero(matrizX);
	gsl_matrix_set_zero(matrizAUX);

    for(i=0;i<filas;i++){
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizX, i, j, matriz_X[i][j]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<columnas;i++) {
		for(j=intervalo1;j<intervalo2;j++){
			gsl_matrix_set(matrizINV, h, t, matriz_INV[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizX, matrizINV, 0.0, matrizAUX); 

    h = 0;
    t = 0;
	for(i=0;i<filas;i++) {
		for(j=intervalo1;j<intervalo2;j++){
			matriz_AUX[i][j] = gsl_matrix_get(matrizAUX,h,t) ;
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizINV);
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizAUX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadSeven(){
    int i,j;
    int t,h;
	gsl_matrix * matrizINV = gsl_matrix_alloc(columnas, intervalo1);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizAUX = gsl_matrix_alloc(filas, intervalo1);
	gsl_matrix_set_zero(matrizINV);
	gsl_matrix_set_zero(matrizX);
	gsl_matrix_set_zero(matrizAUX);

    for(i=0;i<filas;i++){
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizX, i, j, matriz_X[i][j]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<columnas;i++){
		for(j=intervalo2;j<intervalo3;j++){
			gsl_matrix_set(matrizINV, h, t, matriz_INV[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizX, matrizINV, 0.0, matrizAUX); 

    h = 0;
    t = 0;
	for(i=0;i<filas;i++) {
		for(j=intervalo2;j<intervalo3;j++){
			matriz_AUX[i][j] = gsl_matrix_get(matrizAUX,h,t) ;
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizINV);
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizAUX);
	pthread_exit(0);
}

//***************************************************************************************************************

void *threadEight(){
    int i,j;
    int t,h;
	gsl_matrix * matrizINV = gsl_matrix_alloc(columnas, columnas - intervalo3);
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizAUX = gsl_matrix_alloc(filas, columnas - intervalo3);
	gsl_matrix_set_zero(matrizINV);
	gsl_matrix_set_zero(matrizX);
	gsl_matrix_set_zero(matrizAUX);

    for(i=0;i<filas;i++){
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizX, i, j, matriz_X[i][j]);
		}
    }

    h = 0;
    t = 0;
    
    for(i=0;i<columnas;i++) {
		for(j=intervalo3;j<columnas;j++){
			gsl_matrix_set(matrizINV, h, t, matriz_INV[i][j]);
			t++;
		}
		h++;
		t=0;
	}
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizX, matrizINV, 0.0, matrizAUX); 

    h = 0;
    t = 0;
	for(i=0;i<filas;i++) {
		for(j=intervalo3;j<columnas;j++){
			matriz_AUX[i][j] = gsl_matrix_get(matrizAUX,h,t) ;
			t++;
		}
		h++;
		t=0;
	}
	gsl_matrix_free(matrizINV);
	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizAUX);
	pthread_exit(0);
}

//***************************************************************************************************************





int main (){

	intervalo1=75;
	intervalo2=150;
	intervalo3=225;

    FILE *Archivo;
	// GENERAR LA MATRIZ X CON LOS 1 Y LA MATRIZ Y
    Archivo = fopen("prueba.txt","r");
    int i,j;
    if(Archivo==NULL)
        printf("error");
    for(i=0;i<filas;i++){
		fscanf(Archivo, "%lf ", &matriz_Y[i][0]);
		//printf("%.2f \n", matriz_Y[i][0]);
        for(j=0;j<columnas;j++){
			if(j == 0){
				matriz_X[i][j] = 1;
				//printf("%.2f ", matriz_X[i][j]);
			}
			else{
				fscanf(Archivo, "%lf ", &matriz_X[i][j]); //se guarda en un array
				//printf("%.2f  ", matriz_X[i][j]);  //  y se imprime a la vez (aprovechamos por que el bucle es el mismo)
			}
        }
        //printf("\n");      //cada vez que se termina una fila hay que pasar a la siguiente linea
    }
    fclose(Archivo);
    void *exit_status;   
   	pthread_t *thread;

	thread = calloc(8,sizeof(pthread_t));

//	Se declaran dos matrices de 2x2
	gsl_matrix * matrizX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizY = gsl_matrix_alloc(filas, 1);
	gsl_matrix * matrizXT = gsl_matrix_alloc(columnas, filas);
	gsl_matrix * matrizB = gsl_matrix_alloc(columnas, 1);
	gsl_matrix * matrizIN = gsl_matrix_alloc(columnas, columnas);
	gsl_matrix * matrizXTX = gsl_matrix_alloc(columnas, columnas);
	gsl_matrix * matrizXTY = gsl_matrix_alloc(columnas, 1);
	gsl_matrix * matrizINXTY = gsl_matrix_alloc(columnas, 1);
	gsl_matrix * matrizAUX = gsl_matrix_alloc(filas, columnas);
	gsl_matrix * matrizCIR = gsl_matrix_alloc(filas, 1);


	//gsl_matrix * b = gsl_matrix_alloc(2, 2);
	//  Los espacios de memoria de las matrices se hacen 0	
	gsl_matrix_set_zero(matrizX);
	gsl_matrix_set_zero(matrizY);
	gsl_matrix_set_zero(matrizB);
	gsl_matrix_set_zero(matrizXT);
	gsl_matrix_set_zero(matrizXTX);
	gsl_matrix_set_zero(matrizXTY);
	gsl_matrix_set_zero(matrizINXTY);
	gsl_matrix_set_zero(matrizAUX);
	gsl_matrix_set_zero(matrizCIR);

  	struct timeval t_ini, t_fin;
  	double secs;
  	gettimeofday(&t_ini, NULL);

	for(i=0;i<filas;i++) {
		gsl_matrix_set(matrizY, i, 0, matriz_Y[i][0]);
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizX, i, j, matriz_X[i][j]);
		}
	}
    for(i=0;i<columnas;i++){
		for(j=0;j<filas;j++){
			//matriz_X_T[i][j] = matriz_X[j][i];
			gsl_matrix_set(matrizXT, i, j, matriz_X[j][i]);
		}
    }

	pthread_create(&thread[0],NULL,threadOne,NULL);
    pthread_create(&thread[1],NULL,threadTwo,NULL);
    pthread_create(&thread[2],NULL,threadThree,NULL);
    pthread_create(&thread[3],NULL,threadFour,NULL);


    for (i = 0; i < 4; i = i + 1)
        pthread_join(thread[i],&exit_status);


	for(i=0;i<columnas;i++) {
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizXTX, i, j, matriz_XTX[i][j]);
		}
	} 

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXT, matrizY, 0.0, matrizXTY); 

    gsl_linalg_cholesky_decomp(matrizXTX);
	gsl_linalg_cholesky_invert(matrizXTX);

	for(i=0;i<columnas;i++) {
		for(j=0;j<columnas;j++){
			matriz_INV[i][j] = gsl_matrix_get(matrizXTX,i,j);
		}
	} 

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizXTX, matrizXTY, 0.0, matrizB);

	pthread_create(&thread[4],NULL,threadFive,NULL);
    pthread_create(&thread[5],NULL,threadSix,NULL);
    pthread_create(&thread[6],NULL,threadSeven,NULL);
    pthread_create(&thread[7],NULL,threadEight,NULL);


    for (i = 4; i < 8; i = i + 1)
        pthread_join(thread[i],&exit_status);


	for(i=0;i<filas;i++) {
		for(j=0;j<columnas;j++){
			gsl_matrix_set(matrizAUX, i, j, matriz_AUX[i][j]);
		}
	}


	double SCM=0;
  	double SCT=0;
  	double SCR=0;
  	int GL_Modelo = columnas - 1;
  	int GL_Residuos = filas - GL_Modelo - 1;
  	int GL_Totales = filas - 1;
  	double Media_Modelo;
  	double Media_Residuos;
  	double Coef_F;
  	double y_promedio = 0;


  	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, matrizAUX, matrizXTY, 0.0, matrizCIR); 

    for(i=0;i<filas;i++){
		y_promedio = y_promedio + gsl_matrix_get(matrizY,i,0);
	}

    y_promedio = y_promedio/filas;
	for(i=0;i<filas;i++){
		SCM = SCM + ((gsl_matrix_get(matrizCIR,i,0) - y_promedio)*(gsl_matrix_get(matrizCIR,i,0) - y_promedio));
	}

	for(i=0;i<filas;i++){
		SCR = SCR + (gsl_matrix_get(matrizY,i,0) - gsl_matrix_get(matrizCIR,i,0))*(gsl_matrix_get(matrizY,i,0) - gsl_matrix_get(matrizCIR,i,0));
	}

	SCT = SCM + SCR;

	//gettimeofday(&t_fin, NULL);
	//printf("SCR = %.6f \n", SCR );
	Media_Modelo = SCM/GL_Modelo;
	//printf("%.6f \n", Media_Modelo );
	Media_Residuos = SCR/GL_Residuos;
	//printf("%.6f \n", Media_Residuos );
	Coef_F = Media_Modelo/Media_Residuos;
	//printf("%.6f \n", Coef_F );
	//secs = timeval_diff(&t_fin, &t_ini);
	//printf("%.3g Segundos\n", secs);
	gettimeofday(&t_fin, NULL);
  	secs = timeval_diff(&t_fin, &t_ini);
	printf("Fuente de Variacion           SC          G.L.         Media           Cociente F\n");
	printf("     Modelo              %lf       %.i         %lf        %lf \n",SCM, GL_Modelo, Media_Modelo, Coef_F);
	printf(" Residuos/Errores        %lf       %.i         %lf\n", SCR, GL_Residuos, Media_Residuos);
    printf("      Total              %lf       %.i\n", SCT, GL_Totales);

    printf("\n");
    printf("\n");
    printf("%.3g Segundos: \n", secs);

	gsl_matrix_free(matrizX);
	gsl_matrix_free(matrizY);
	gsl_matrix_free(matrizXT);
	gsl_matrix_free(matrizB);
	gsl_matrix_free(matrizXTX);
	gsl_matrix_free(matrizXTY);
	gsl_matrix_free(matrizINXTY);
	gsl_matrix_free(matrizAUX);
	gsl_matrix_free(matrizCIR);

	return 0;
}