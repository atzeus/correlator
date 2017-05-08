#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define NR_STATIONS_PER_BLOCK                 16
#define NR_TIMES_PER_BLOCK                    8
#define NR_BLOCK_X             (NR_RECEIVERS / NR_STATIONS_PER_BLOCK)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE
#define SIMD_WORK_ITEMS                8

typedef signed char int8_t;
typedef float2 fcomplex;
typedef float16 fcomplex8;

typedef fcomplex8 InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS/8]/*[NR_CHANNELS]*/;
typedef fcomplex8 OutputType[NR_BLOCKS][NR_STATIONS_PER_BLOCK]/*[NR_CHANNELS]*/; 


fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

__kernel __attribute__((reqd_work_group_size(NR_STATIONS_PER_BLOCK,NR_STATIONS_PER_BLOCK,1))) 
				 __attribute__((num_simd_work_items(SIMD_WORK_ITEMS)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
  uint block  = get_group_id(0);
  uint blockX = (unsigned) (sqrt(convert_float(8 * block + 1)) - 0.99999f) / 2;
  uint blockY = block - blockX * (blockX + 1) / 2;

  uint firstStationX = (blockX + 1) ;
  uint firstStationY = blockY ;

	uint loadTime = get_local_id(0);
	uint staty = get_local_id(0);


  __local fcomplex8 memx[NR_TIMES_PER_BLOCK];
  __local fcomplex memy[NR_TIMES_PER_BLOCK][NR_STATIONS_PER_BLOCK];
  fcomplex8 sum = (fcomplex8)0;
	#pragma unroll 1
	for (int major = 0 ; major < NR_SAMPLES_PER_CHANNEL; major+=NR_TIMES_PER_BLOCK ){
		barrier(CLK_LOCAL_MEM_FENCE);
		memx[loadTime] = (*input)[major + loadTime ][firstStationX];
		fcomplex8 ready = (*input)[major + loadTime ][firstStationY];
		memy[loadTime][0] = (float2)(ready.s0,ready.s1);
		memy[loadTime][1] = (float2)(ready.s2,ready.s3);
		memy[loadTime][2] = (float2)(ready.s4,ready.s5);
		memy[loadTime][3] = (float2)(ready.s6,ready.s7);
		memy[loadTime][4] = (float2)(ready.s8,ready.s9);
		memy[loadTime][5] = (float2)(ready.sA,ready.sB);
		memy[loadTime][6] = (float2)(ready.sC,ready.sD);
		memy[loadTime][7] = (float2)(ready.sE,ready.sF);
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll 1
		for(int time = 0 ; time < NR_TIMES_PER_BLOCK; time++){
			fcomplex b = memy[time][staty];
			sum.s0 += memx[time].s0 * b.x + memx[time].s1 * b.y;
			sum.s1 += memx[time].s1 * b.x - memx[time].s0 * b.y;

			sum.s2 += memx[time].s2 * b.x + memx[time].s3 * b.y;
			sum.s3 += memx[time].s3 * b.x - memx[time].s2 * b.y;

			sum.s4 += memx[time].s4 * b.x + memx[time].s5 * b.y;
			sum.s5 += memx[time].s5 * b.x - memx[time].s4 * b.y;

			sum.s6 += memx[time].s6 * b.x + memx[time].s7 * b.y;
			sum.s7 += memx[time].s7 * b.x - memx[time].s6 * b.y;

			sum.s8 += memx[time].s8 * b.x + memx[time].s9 * b.y;
			sum.s9 += memx[time].s9 * b.x - memx[time].s8 * b.y;

			sum.sA += memx[time].sA * b.x + memx[time].sB * b.y;
			sum.sB += memx[time].sB * b.x - memx[time].sA * b.y;

			sum.sC += memx[time].sC * b.x + memx[time].sD * b.y;
			sum.sD += memx[time].sD * b.x - memx[time].sC * b.y;

			sum.sE += memx[time].sE * b.x + memx[time].sF * b.y;
			sum.sF += memx[time].sF * b.x - memx[time].sE * b.y;
		barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	(*output)[block][staty] = sum;
 }



