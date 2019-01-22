#include <stdio.h>
#include <stdlib.h>

#define DATA_ROWS 28
#define DATA_COLUMNS 28

#define CONV_1_N_KERNELS 5
#define CONV_1_KERNEL_ROWS 3
#define CONV_1_KERNEL_COLUMNS 3
#define CONV_1_KERNEL_SIZE (CONV_1_KERNEL_ROWS-1)/2

#define POOL_1_KERNEL_ROWS 2
#define POOL_1_KERNEL_COLUMNS 2

#define KERNEL_ROWS 3
#define KERNEL_COLUMNS 3

#define OUTPUT_ROWS DATA_ROWS-CONV_1_KERNEL_SIZE*2
#define OUTPUT_COLUMNS DATA_COLUMNS-CONV_1_KERNEL_SIZE*2


#define CONV_1_OUTPUT_ROWS DATA_ROWS-CONV_1_KERNEL_SIZE*2
#define CONV_1_OUTPUT_COLUMNS DATA_COLUMNS-CONV_1_KERNEL_SIZE*2
#define MAXPOOL_1_OUTPUT_ROWS CONV_1_OUTPUT_ROWS/2
#define MAXPOOL_1_OUTPUT_COLUMNS CONV_1_OUTPUT_COLUMNS/2




float data[DATA_ROWS][DATA_COLUMNS];			// Input data

float output[OUTPUT_ROWS][OUTPUT_COLUMNS];		// Temporary output for conv_1
float maxpool_1_temp[MAXPOOL_1_OUTPUT_ROWS][MAXPOOL_1_OUTPUT_COLUMNS];

float conv_1_output[CONV_1_N_KERNELS][OUTPUT_ROWS][OUTPUT_COLUMNS];		// Output for conv_1
float maxpool_1_output[CONV_1_N_KERNELS][MAXPOOL_1_OUTPUT_ROWS][MAXPOOL_1_OUTPUT_ROWS];		// Output for maxpool_1

float conv_1_kernels[CONV_1_N_KERNELS][CONV_1_KERNEL_ROWS][CONV_1_KERNEL_COLUMNS] = {
		{
			{-0.1905201, 0.15129338,0.34179518},
			{ 0.30804652,0.5119313, 0.36381802},
			{ 0.5456993, 0.10510398,0.19406371}
		},
		{
			{-0.13744535,-0.18738753,-0.14871328},
			{ 0.16939132,-0.22470851,-0.25093016},
			{ 0.1424438,-0.044466432,-0.048779003}
		},
		{
			{ 0.5324741,0.6786809,0.7371071},
			{ 0.078251556,0.17476529,0.099192075},
			{-0.21780658,-0.3565439,-0.010590944}
		},
		{
			{ 0.107310444,-0.23381963,0.26083875},
			{ 0.07189208,-0.27035508,-0.14091091},
			{ 0.2087852,-0.31711155,-0.107295476}
		},
		{
			{-0.25686505,-0.23432541,0.33097032},
			{-0.08210149,-0.031552035,0.36062017},
			{ 0.45140123,0.34630477,0.5247166}
}};

float conv_1_biases[CONV_1_N_KERNELS] = {0.3351, -0.3321, -0.0860, -0.0881, -0.0182};

void load_data(void) {
	// Loads input example into data array
	FILE *myFile;
	myFile = fopen("mnist_example2.csv", "r");

	for (int row = 0; row < DATA_ROWS; row++) {
		for (int column = 0; column < DATA_COLUMNS; column++) {
			fscanf(myFile, "%f,", &data[row][column] );
		}
	}
}

int check_data(float data[26][26]) {
	printf("Diagonal:\n");
	for (int row = 0; row < 26; row++ ) {
			for (int column = 0; column < 26; column++ ) {
				if (row == column) {
					printf("%.2f ", data[row][column]);
				}
			}
	}
	printf("\n");
}

int show_data(float data[DATA_ROWS][DATA_COLUMNS]) {
	printf("Data:\n");
	for (int row = 0; row < DATA_ROWS; row++ ) {
			for (int column = 0; column < DATA_COLUMNS; column++ ) {
				printf("%.2f ", data[row][column]);
			}
			printf("\n");
	}
}

int conv_1(float image[DATA_ROWS][DATA_COLUMNS], float kernel[CONV_1_KERNEL_ROWS][CONV_1_KERNEL_COLUMNS]) {
	for (int row = 0; row < DATA_ROWS; row++) {
		for (int column = 0; column < DATA_COLUMNS; column++) {
			float total = 0;
			for (int kernel_row = 0; kernel_row < CONV_1_KERNEL_ROWS; kernel_row++) {
				for (int kernel_column = 0; kernel_column < CONV_1_KERNEL_COLUMNS; kernel_column++) {
					total += image[row+kernel_row][column+kernel_column] * kernel[kernel_row][kernel_column];
				}
			}
			output[row][column] = total;
		}
	}
}

int sconv_1() {
	for (int i = 0; i < 1 ; i++) { 		// For every image in batch (for now its just one)
		for (int j = 0; j < 5; j++) { 	// For every kernel
			float result = 0;
			conv_1(data, conv_1_kernels[j]);

			for (int row=0;row<CONV_1_OUTPUT_ROWS;row++) { // Copies temporary results into conv_1_output memory
				for (int col = 0;col<CONV_1_OUTPUT_COLUMNS;col++) {
					conv_1_output[j][row][col] = output[row][col] + conv_1_biases[j];
				}
			}
		}
	}

	return 0;
}

int maxpool_1(float image[CONV_1_OUTPUT_ROWS][CONV_1_OUTPUT_COLUMNS]) {
	float temp[4];

	for (int row = 0; row < MAXPOOL_1_OUTPUT_ROWS; row++) {
		for (int column= 0; column< MAXPOOL_1_OUTPUT_COLUMNS; column++) {
			// Temporary values set to 0 means that this pooling also does relu!
			float temp[4] = {0.0, 0.0, 0.0, 0.0};
			float max_value = 0;
			temp[0] = image[row*2][column*2];
			temp[1] = image[row*2+1][column*2];
			temp[2] = image[row*2][column*2+1];
			temp[3] = image[row*2+1][column*2+1];

			for (int i = 0; i < 4; i++) { // Find max value
				if (temp[i] >= max_value) {
					max_value = temp[i];
				}
				//printf("%f\n", max_value);

			}
			maxpool_1_temp[row][column] = max_value;
		}
	}
}

int spool_1() {
	for (int i = 0; i < CONV_1_N_KERNELS; i++) {	// Runs pooling for every conv_1 output product
		maxpool_1(conv_1_output[i]);

		for (int row=0;row<MAXPOOL_1_OUTPUT_ROWS;row++) { // Copies temporary results into maxpool_1_output memory
			for (int col = 0;col<MAXPOOL_1_OUTPUT_COLUMNS;col++) {
				maxpool_1_output[i][row][col] = maxpool_1_temp[row][col];
			}
		}
	}
}

int show_output(float output[OUTPUT_ROWS][OUTPUT_COLUMNS]) {
	printf("Output:\n");
	for (int row = 0; row < OUTPUT_ROWS; row++) {
		for (int column = 0; column < OUTPUT_COLUMNS; column++) {
				printf("%.2f ", output[row][column]);
			}
			printf("\n");
	}
}

int main() {
	printf("System ready!\n\r");
	load_data();
	//show_data(data);
	sconv_1();
	printf("Conv_1 check: %f\n\r", conv_1_output[2][3][4]);
	maxpool_1(conv_1_output[0]);
	printf("Maxpool_1 check: %f\n\r", maxpool_1_temp[9][9]);
	spool_1();
	printf("Maxpool_1 check: %f\n\r", maxpool_1_output[2][9][9]);



	return 0;
}
