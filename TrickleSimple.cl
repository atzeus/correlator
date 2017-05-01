#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define BLOCK_SIZE             16
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                ((NR_RECEIVERS * (NR_RECEIVERS + 1)) / 2)

typedef signed char int8_t;
typedef float2 fcomplex;

typedef fcomplex InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS]/*[NR_CHANNELS]*/;
typedef fcomplex OutputType[NR_BLOCKS][BLOCK_SIZE][BLOCK_SIZE]/*[NR_CHANNELS]*/; 


fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
	int blockx = 0;
	int blocky = 0;
	#pragma unroll 1
	for( int baselineBlock = 0 ; baselineBlock < NR_BLOCKS; baselineBlock++) {
	    fcomplex sums[BLOCK_SIZE][BLOCK_SIZE];
 		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE ; i++){
	 		#pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE ; j++){
				sums[i][j] = (fcomplex)0;
			}
		}
		fcomplex bufx[2 * BLOCK_SIZE - 1][ BLOCK_SIZE ];
		#pragma unroll
		for(int i = 0 ; i < 2 * BLOCK_SIZE - 1 ; i++){
	 		#pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE  ; j++){
				bufx[i][j] = (fcomplex)0;
			}
		}
		fcomplex bufy[BLOCK_SIZE ][2 * BLOCK_SIZE - 1];
		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE ; i++){
	 		#pragma unroll
			for(int j = 0 ; j < 2 * BLOCK_SIZE - 1  ; j++){
				bufy[i][j] = (fcomplex)0;
			}
		}
		#pragma unroll 1
		for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){
			#pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE ; j++){	 	
				#pragma unroll
				for(int i = 2 * BLOCK_SIZE - 2 ; i >=1 ; i--){				
					bufx[i][j] = bufx[i-1][j];
				}
			}
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE ; i++){	 	
				#pragma unroll
				for(int j = 2 * BLOCK_SIZE - 2 ; j >=1 ; j--){				
					bufy[i][j] = bufy[i][j-1];
				}
			}
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE ; i++){	 	
				bufx[BLOCK_SIZE - 1 -  i][i] =  (*input)[time][blockx + i];
			}
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE ; i++){	 	
				bufy[i][BLOCK_SIZE - 1 - i] =  (*input)[time][blocky + i];
			}
			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE ; i++){
				#pragma unroll
				for(int j = 0; j < BLOCK_SIZE ; j++){
					float2 a = bufx[i + BLOCK_SIZE - 1][j];
					float2 b = bufy[i][j + BLOCK_SIZE - 1];
					sums[i][j] +=  (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
				}
			}
		}
		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE ; i++){
		  #pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE ; j++){
					(*output)[baselineBlock][i][j] = sums[i][j];
			}
		}

		blocky+=BLOCK_SIZE;
		if(blocky > blockx){
			blockx+=BLOCK_SIZE;
			blocky = 0;
		}
  }
}


