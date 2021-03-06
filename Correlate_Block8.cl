#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define BLOCK_SIZE_X             16
#define BLOCK_SIZE_Y            (BLOCK_SIZE_X / 2)
#define BLOCK_SIZE              (BLOCK_SIZE_X * BLOCK_SIZE_X)
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE_X)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                ((NR_RECEIVERS * (NR_RECEIVERS + 1)) / 2)

typedef signed char int8_t;
typedef float2 fcomplex;

typedef fcomplex InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS]/*[NR_CHANNELS]*/;
// Write non-contigous: 
// typedef fcomplex OutputType[OUTSIZE]/*[NR_CHANNELS]*/; 
// Write contigous:
typedef fcomplex OutputType[NR_BLOCKS][BLOCK_SIZE_X][BLOCK_SIZE_X]/*[NR_CHANNELS]*/; 


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
	  fcomplex sums[BLOCK_SIZE_X][BLOCK_SIZE_X];
 		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
	 		#pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE_X ; j++){
				sums[i][j] = (fcomplex)0;
			}
		}
		#pragma unroll 1
		for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){
			fcomplex memx[BLOCK_SIZE_X];
			fcomplex memy[BLOCK_SIZE_X];
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
				memx[i] = (*input)[time][blockx + i];
			}
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
				memy[i] = (*input)[time][blocky + i];
			}
					#pragma unroll				
					for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
						memx[i] = memxnew[i];
					}
				} else {
					if(time!= NR_SAMPLES_PER_CHANNEL - 1){
						#pragma unroll					
						for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
							memxnew[i] = (*input)[time+1][blockx + i];
						}
					}
				}
				#pragma unroll
				for(int i = 0; i < BLOCK_SIZE_X ; i++){
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE_Y ; j++){
						float2 a = memx[i];
						float2 b = memy[z * BLOCK_SIZE_Y + j];
						lsums[z][i][j] =  (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
					}
				}
			}
			#pragma unroll
			for(int z = 0 ; z < 2 ; z++){
				#pragma unroll
				for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
				  #pragma unroll
					for(int j = 0 ; j < BLOCK_SIZE_Y ; j++){
						sums[i][z * BLOCK_SIZE_Y + j] += lsums[z][i][j];
					}
				}
			}
		}
		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
		  #pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE_X ; j++){
					(*output)[baselineBlock][i][j] = sums[i][j];
			}
		}

		blocky+=BLOCK_SIZE_X;
		if(blocky > blockx){
			blockx+=BLOCK_SIZE_X;
			blocky = 0;
		}
  }
}


