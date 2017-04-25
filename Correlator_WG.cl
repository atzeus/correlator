#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define BLOCK_SIZE_X                 8
#define BLOCK_SIZE              (BLOCK_SIZE_X * BLOCK_SIZE_X)
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE_X)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE
#define SIMD_WORK_ITEMS                4

typedef signed char int8_t;
typedef float2 fcomplex;

typedef fcomplex InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS]/*[NR_CHANNELS]*/;
typedef fcomplex OutputType[OUTSIZE]/*[NR_CHANNELS]*/; 


fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE_X,BLOCK_SIZE_X,1))) 
				 __attribute__((num_simd_work_items(SIMD_WORK_ITEMS)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
  uint block  = get_group_id(0);
  uint blockx = (unsigned) (sqrt(convert_float(8 * block + 1)) - 0.99999f) / 2;
  uint blocky = block - blockx * (blockx + 1) / 2;

	uint x = get_local_id(0);
	uint y = get_local_id(1);

  fcomplex res = (fcomplex)0;
  __local fcomplex memx[BLOCK_SIZE_X];
  __local fcomplex memy[BLOCK_SIZE_X];

	for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){
		memx[x] = (*input)[time][blockx + x];
		memy[y] = (*input)[time][blocky + y];
		res += mulConj( memx[x] , memy[y]);
	}
	(*output)[block + x * BLOCK_SIZE + y] = res;
 }



