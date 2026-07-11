#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#define APRENDIZAJE 0.01 //Esta sera la tasa de aprendizaje, no puede ser ni muy alto ni muy bajo, ya que marcara de tamaño de los saltos de aprendizaje
#define TAMANO 784 //Tamaño de la imagen, ya que es de 28*28 pixeles
#define IMAGENES 60000

//Funcion auxiliar con CUDA que aplica los pesos correspondientes segun los pixeles de la imagen
__global__ void AplicarPesos (float* ValorDigitos, float* Pesos, unsigned char * Imagen, int tamano, int actual){
	//Lanzamos 10 bloques, uno para cada digito, con 784 hilos, uno para cada pixel de su respectiva imagen
	int pixel = threadIdx.x;
	int digito =  blockIdx.x;
	__shared__ float sumatorioCompartido[TAMANO]; //Usamos memoria compartida en cada bloque, asi cada hilo puede ver la memoria de todo el bloque al que pertenece
	if (pixel < tamano){
		//Cada hilo hace su propio calculo
		int indicePeso = digito*tamano+pixel;
		int indiceImagen = actual*tamano+pixel;
		float pixelNormalizado = (float)Imagen[indiceImagen]/255.0f;
		sumatorioCompartido[pixel] = Pesos[indicePeso] * pixelNormalizado;
	} else sumatorioCompartido[pixel] = 0.0f; //Para controlar el error de lanzar mas hilos
	
	__syncthreads(); //Esperamos que todos los hilos lleguen al mismo punto
	
	//Hacemos una suma en arbol
	for(int i = blockDim.x/2; i > 0; i >>=1) {
		if(pixel<i && (pixel+i) < tamano) sumatorioCompartido[pixel] += sumatorioCompartido[pixel+i];
		__syncthreads();
	}

	//El hilo 0 de cada bloque tendra el sumatorio total
	if(pixel == 0){
		//Debemos aplicar la aplicacion sigmoide, f(x) = 1/(1+e^-x)
		float sumatorio = sumatorioCompartido[0] * -1.0f; //Le cambiamos el signo para la exponencial
		ValorDigitos[digito] = __fdividef(1.0f, 1.0f + __expf(sumatorio)); //Usamos los intrinsecos de la gpu
	}
}

//Funcionn auxiliar con CUDA para corregir los pesos de todos los digitos tras comparar con el correcto
__global__ void CorregirPesos(float * ValorDigitos, float * Pesos, unsigned char * Imagen, unsigned char * respuesta, float Aprendizaje, int tamano, int actual){
	//Lanzamos los mismos bloques e hilos que en el anterior kernel
	int digito = blockIdx.x;
	int pixel = threadIdx.x;
	if(digito < 10){
		float error = (digito == respuesta[actual]) ? ValorDigitos[digito]-1 : ValorDigitos[digito]; // El error no sera el mismo para el digito correcto en comparacion con un digito distinto, debemos diferenciar el caso
		float pixelNormalizado = (float)Imagen[actual*tamano+pixel]/255.0f;
		Pesos[digito*tamano+pixel] -= (Aprendizaje*error*pixelNormalizado);
	}
}


//Funcion auxiliar para calcular el digito mas probable segun los pesos anteriormente calculados
unsigned char MasProbable(float * ValorDigitos){
	float Probable = -1;
	unsigned char Devolver = 0;
	unsigned char indice = 0;
	while(indice < 10){
		if (Probable < ValorDigitos[indice]){
			Probable = ValorDigitos[indice];
			Devolver = indice;
		}
		indice++;
	}
	return Devolver;
}

void MostrarProgreso(int actual){
	float porcentaje = ((float)actual/IMAGENES)*100;
	int anchoBarra = 30;
	int posicion = (actual * anchoBarra)/IMAGENES;

	printf("\rEntrenando: [");
	for(int i = 0; i < anchoBarra; i++){
		if(i < posicion) printf("#");
		else if(i == posicion) printf(">");
		else printf(" ");
	}

	printf("] %.2f%%, (%d/%d)", porcentaje, actual, IMAGENES);
	fflush(stdout);
}



int main(void){

	//Creamos las variables
	srand(time(NULL)); //Inicializamos el generador aleatorio de numeros
	float ValorDigitos[10]; //Guardara el procentaje de probabilidad de cada uno de los digitos
	float Pesos[10*TAMANO]; //Guardara los pesos de cada uno de los 784 pixeles de la imagen para cada uno de los digitos, lo almacenamos en una matriz aplanada para poder usar la gpu
	static unsigned char Imagen[TAMANO*IMAGENES]; //Guardara el valor de los pixeles de la imagen original, debe ser static para no desbordar el tamaño preestablecido por linux de 8MiB, o usar memoria dinamica con malloc
	unsigned char respuesta[IMAGENES]; //Guardara la respuesta real de la imagen analizada, lo necesitamos para comparar
	
	//Abrimos el archivo de entrenamiento, lo abrimos en lectura binaria
	FILE * Imagenes =  fopen("train-images-idx3-ubyte", "rb");
	//Hacemos exactamente lo mismo con el de las respuestas
	FILE * Respuestas = fopen("train-labels-idx1-ubyte", "rb");

	if (Respuestas != NULL && Imagenes !=  NULL){
		//Debemos saltarnos las cabeceras de ambos archivos, en el primero tenemos una de 16 Bytes, en el segundo es de 8 Bytes
		fseek(Imagenes, 16, SEEK_SET);
		fseek(Respuestas, 8, SEEK_SET);

		//Leemos todas las imagenes y respuestas de golpe
		fread(Imagen, sizeof(unsigned char), TAMANO*IMAGENES, Imagenes);
		fread(respuesta, sizeof(unsigned char), IMAGENES, Respuestas);

		//Ya podemos cerrar los archivos
		fclose(Imagenes);
		fclose(Respuestas);

		//Sabemos que el archivo de entrenamiento tiene un total de 60000 imagenes, asi que usaremos un contador hasta 60000 para marcar el final del bucle
		int limite = 0;

		//Hasta ahora no parece haber ningun tipo de problema, asi que podemos inicializar los pesos, como no podemos asignar unos concretos, lo haremos con rand(), que asignara un numero aleatorio
		for(int i = 0; i < 10*TAMANO; i++) Pesos[i] = ((float)rand()/RAND_MAX) * 0.2f - 0.1f;

		//Inicializamos las probabilidades, por seguridad
		for(int i = 0; i<10; i++) ValorDigitos[i] = 0;

		//Debemos recordar que la gpu no tiene acceso directo a la memoria de la cpu, por tanto, debemos reservar memoria explicitamente en la gpu para las mismas variables
		unsigned char * dImagen;
		cudaMalloc((void**)&dImagen, TAMANO*IMAGENES*sizeof(unsigned char));
		float * dValorDigitos;
		cudaMalloc((void**)&dValorDigitos, 10*sizeof(float));
		cudaMemcpy(dValorDigitos,ValorDigitos, 10*sizeof(float), cudaMemcpyHostToDevice); 
		unsigned char * dRespuestas;
		cudaMalloc((void**)&dRespuestas, IMAGENES*sizeof(unsigned char));
		float * dPesos;
		cudaMalloc((void**)&dPesos, 10*TAMANO*sizeof(float));
		cudaMemcpy(dPesos, Pesos, 10*TAMANO*sizeof(float), cudaMemcpyHostToDevice);

		//Le pasamos todos los datos de la cpu a la gpu
		cudaMemcpy(dImagen, Imagen, TAMANO*IMAGENES*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(dRespuestas, respuesta, IMAGENES*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(dValorDigitos, ValorDigitos, 10*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dPesos, Pesos, 10*TAMANO*sizeof(float), cudaMemcpyHostToDevice);

		while (limite < IMAGENES){

			//Aplicamos los pesos a los pixeles de la imagen, para eso llamamos a la funcion AplicarPesos que se ejcutara directamente en la gpu
			AplicarPesos<<<10, TAMANO>>>(dValorDigitos, dPesos, dImagen, TAMANO, limite); //Esta funcion, en lugar de ser de 10*784 iteraciones (es decir, 7840 iteraciones), sera mucho menor debido a la programacion paralela que ofrece cuda
			cudaDeviceSynchronize(); //Por defecto, la cpu tras llamar a un kernel de CUDA, continuara con la siguiente linea, al ser una ejecucion asincrona, asi que obligamos a que se espere
		
			//Recuperamos las probabilidades de cada digito. Estan en la GPU y los necesitamos en la CPU
			cudaMemcpy(ValorDigitos, dValorDigitos, 10*sizeof(float), cudaMemcpyDeviceToHost);
			
			//Una vez aplicados los pesos, comprobamos el digito mas probable
			unsigned char Probable = MasProbable(ValorDigitos);
			printf("El digito de la imagen es %hhu, con una seguridad del %f; ", Probable, ValorDigitos[Probable]*100);
			printf("La respuesta real era %hhu \n", respuesta[limite]);

			//Corregimos los pesos, necesitamos simplemente un 10 bloques de 784 hilos, igual que en el caso anterior
			CorregirPesos<<<10, TAMANO>>>(dValorDigitos, dPesos, dImagen, dRespuestas, APRENDIZAJE, TAMANO, limite);
			cudaDeviceSynchronize();
			
			//MostrarProgreso(limite); //Imprimimos por pantalla el progreso del entrenamiento
			limite++;
		}
		
		cudaDeviceSynchronize(); //No podemos liberar datos de la gpu si aun no hemos terminado de procesarlaos en lineas anteriores

		//Liberamos la memoria de la GPU
		cudaFree(dImagen);
		cudaFree(dValorDigitos);
		cudaFree(dPesos);
	}

	return 0;
}

