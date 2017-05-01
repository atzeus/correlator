#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define NR_7                    2
#define BLOCK_SIZE_X            (7 * NR_7)
#define BLOCK_SIZE              (BLOCK_SIZE_X * BLOCK_SIZE_X)
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE_X)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE

typedef signed char int8_t;

typedef ulong2 InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS/7]/*[NR_CHANNELS]*/;
typedef int2 OutputType[NR_BLOCKS][BLOCK_SIZE_X][BLOCK_SIZE_X]/*[NR_CHANNELS]*/; 



int2  __attribute__((__overloadable__,__always_inline__,const)) mulConj(ushort2 a, ushort2 b){
  return (int2)(0x7FFFF & 
								((0x3FFFF & ((0x1FF & a.x) * (0x1FF & b.x))) + 
								 (0x3FFFF & ((0x1FF & a.y) * (0x1FF & b.y))))
			, (0x7FFFF & 
								((0x3FFFF & ((0x1FF & a.y) * (0x1FF & b.x))) -
								 (0x3FFFF & ((0x1FF & a.x) * (0x1FF & b.y))))));
}

__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
	int blockx = 0;
	int blocky = 0;

	for( int baselineBlock = 0 ; baselineBlock < NR_BLOCKS; baselineBlock++) {
    int2 sums[BLOCK_SIZE_X][BLOCK_SIZE_X];
 		#pragma unroll
		for(int i = 0 ; i < BLOCK_SIZE_X ; i++){
	 		#pragma unroll
			for(int j = 0 ; j < BLOCK_SIZE_X ; j++){
				sums[i][j] = (int2)0;
			}
		}
		for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){

			short2 memx[BLOCK_SIZE_X] ;
			#pragma unroll
      for(int m = 0 ; m < NR_7 ; m++){
				ulong2 a = (*input)[time][blockx + m]; 
				#pragma unroll
				for(int i = 0 ; i < 7 ; i++){
					memx[m * 7 + i].x = ((a.x & (0x1FF << (i * 9))) >> (i * 9)) & 0x1FF;
				}
				#pragma unroll
				for(int i = 0 ; i < 7 ; i++){
					memx[m * 7 + i].y = ((a.y & (0x1FF << (i * 9))) >> (i * 9)) & 0x1FF;
				}
			}
			short2 memy[BLOCK_SIZE_X];
			#pragma unroll
      for(int m = 0 ; m < NR_7 ; m++){
				
				ulong2 a = (*input)[time][blocky + m]; 
				#pragma unroll
				for(int i = 0 ; i < 7 ; i++){
					memy[m * 7 + i].x = ((a.x & (0x1FF << (i * 9))) >> (i * 9)) & 0x1FF;
				}
				#pragma unroll
				for(int i = 0 ; i < 7 ; i++){
					memy[m * 7 + i].y = ((a.y & (0x1FF << (i * 9))) >> (i * 9)) & 0x1FF;
				}
			}

			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE_X ; i++){
				#pragma unroll
				for(int j = 0; j < BLOCK_SIZE_X ; j++){
					short2 a = memx[i];
					short2 b = memy[j];
					sums[i][j] += (int2)(0x7FFFF & 
								((0x3FFFF & ((0x1FF & a.x) * (0x1FF & b.x))) + 
								 (0x3FFFF & ((0x1FF & a.y) * (0x1FF & b.y))))
			, (0x7FFFF & 
								((0x3FFFF & ((0x1FF & a.y) * (0x1FF & b.x))) -
								 (0x3FFFF & ((0x1FF & a.x) * (0x1FF & b.y))))));
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

		blocky+=NR_7;
		if(blocky > blockx){
			blockx+=NR_7;
			blocky = 0;
		}
  }
}


