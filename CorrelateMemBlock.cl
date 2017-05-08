#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define NR_TIME_BLOCKS        (NR_SAMPLES_PER_CHANNEL / BLOCK_SIZE)
#define BLOCK_SIZE                 16
#define SIMD_SIZE										16
#define LOAD_SIZE										16
#define SIMD_BLOCKS               (BLOCK_SIZE / SIMD_SIZE)
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE

typedef signed char int8_t;
typedef float2 fcomplex;
typedef float16 fcomplex8;

typedef fcomplex InputType[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS];
typedef fcomplex OutputType[NR_CHANNELS][NR_BLOCKS][BLOCK_SIZE][BLOCK_SIZE]; 


fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}


__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
	for(int ch = 0 ; ch < NR_CHANNELS; ch++){
		int blockx = 0;
		int blocky = 0;
		for( int baselineBlock = 0 ; baselineBlock < NR_BLOCKS; baselineBlock++) {
			fcomplex sum[BLOCK_SIZE][BLOCK_SIZE];
			for(int x = 0 ; x < BLOCK_SIZE ; x++){
					for(int y = 0 ; y < BLOCK_SIZE ; y++){
						sum[x][y] = 0;
					}
			}
			for(int timem = 0 ; timem < NR_SAMPLES_PER_CHANNEL ; timem+= BLOCK_SIZE){

				fcomplex localx[BLOCK_SIZE][BLOCK_SIZE];
				fcomplex localy[BLOCK_SIZE][BLOCK_SIZE];
		
				for(int time = 0 ; time < BLOCK_SIZE ; time++){
					#pragma unroll SIMD_SIZE
					for(int i = 0 ; i < BLOCK_SIZE ; i++){
						localx[i][time] = (*input)[ch][timem + time][blockx + i];
						localy[i][time] = (*input)[ch][timem + time][blocky + i];
					}
				}
				for(int x = 0 ; x < BLOCK_SIZE ; x++){
					#pragma unroll SIMD_SIZE
					for(int y = 0 ; y < BLOCK_SIZE ; y++){
						#pragma unroll 
						for(int time = 0 ; time < BLOCK_SIZE ; time++) {
							sum[x][y] += mulConj(localx[x][time], localy[y][time]);
						}
					}
				}

			}
		
			#pragma unroll
				for(int x = 0 ; x < BLOCK_SIZE ; x++){
					for(int y = 0 ; y < BLOCK_SIZE ; y++){
						(*output)[ch][baselineBlock][x][y]= sum[x][y];
					}
			}

			blocky+=1;
			if(blocky > blockx){
				blockx+=1;
				blocky = 0;
			}	
		}
	}
	  
}



