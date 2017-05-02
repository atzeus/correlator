#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define BLOCK_SIZE                 8
#define BLOCK_SIZE_Y               16
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE)           
#define NR_BLOCK_Y             (NR_RECEIVERS / BLOCK_SIZE_Y)          
#define NR_BLOCKS              (NR_BLOCK_Y * (NR_BLOCK_Y + 1)) 
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE

typedef signed char int8_t;
typedef float2 fcomplex;
typedef float16 fcomplex8;

typedef fcomplex8 InputType[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS/8];
typedef fcomplex8 OutputType[NR_CHANNELS][NR_BLOCKS][BLOCK_SIZE_Y]; 

/*
fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}
*/

__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
	for(int ch = 0 ; ch < NR_CHANNELS; ch++){
		int blockx = -1;
		int blocky = 0;
		

		for( int baselineBlock = 0 ; baselineBlock < NR_BLOCKS; baselineBlock++) {
			fcomplex8 rowmem[NR_SAMPLES_PER_CHANNEL];
			if(blocky > blockx){
				blockx+=1;
				blocky = 0;
				#pragma unroll
				for(int i = 0 ; i < NR_SAMPLES_PER_CHANNEL ; i++){
					rowmem[i] = (*input)[ch][i][blockx];
				}
			}
			
			fcomplex8 sum[BLOCK_SIZE_Y];
	 		#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE_Y ; i++){
				sum[i] = (fcomplex8)0;
			}
			for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){
			
		  		fcomplex8 memy0 = (*input)[ch][time][blocky];
		  		fcomplex8 memy1 = (*input)[ch][time][blocky + 1];
				sum[0].s0 += rowmem[time].s0*memy0.s0 + rowmem[time].s1*memy0.s1;
				sum[0].s1 += rowmem[time].s1*memy0.s0 + rowmem[time].s0*memy0.s1;

				sum[0].s2 += rowmem[time].s2*memy0.s0 + rowmem[time].s3*memy0.s1;
				sum[0].s3 += rowmem[time].s3*memy0.s0 + rowmem[time].s2*memy0.s1;

				sum[0].s4 += rowmem[time].s4*memy0.s0 + rowmem[time].s5*memy0.s1;
				sum[0].s5 += rowmem[time].s5*memy0.s0 + rowmem[time].s4*memy0.s1;

				sum[0].s6 += rowmem[time].s6*memy0.s0 + rowmem[time].s7*memy0.s1;
				sum[0].s7 += rowmem[time].s7*memy0.s0 + rowmem[time].s6*memy0.s1;

				sum[0].s8 += rowmem[time].s8*memy0.s0 + rowmem[time].s9*memy0.s1;
				sum[0].s9 += rowmem[time].s9*memy0.s0 + rowmem[time].s8*memy0.s1;

				sum[0].sA += rowmem[time].sA*memy0.s0 + rowmem[time].sB*memy0.s1;
				sum[0].sB += rowmem[time].sB*memy0.s0 + rowmem[time].sA*memy0.s1;

				sum[0].sC += rowmem[time].sC*memy0.s0 + rowmem[time].sD*memy0.s1;
				sum[0].sD += rowmem[time].sD*memy0.s0 + rowmem[time].sC*memy0.s1;

				sum[0].sE += rowmem[time].sE*memy0.s0 + rowmem[time].sF*memy0.s1;
				sum[0].sF += rowmem[time].sF*memy0.s0 + rowmem[time].sE*memy0.s1;

				sum[1].s0 += rowmem[time].s0*memy0.s2 + rowmem[time].s1*memy0.s3;
				sum[1].s1 += rowmem[time].s1*memy0.s2 + rowmem[time].s0*memy0.s3;

				sum[1].s2 += rowmem[time].s2*memy0.s2 + rowmem[time].s3*memy0.s3;
				sum[1].s3 += rowmem[time].s3*memy0.s2 + rowmem[time].s2*memy0.s3;

				sum[1].s4 += rowmem[time].s4*memy0.s2 + rowmem[time].s5*memy0.s3;
				sum[1].s5 += rowmem[time].s5*memy0.s2 + rowmem[time].s4*memy0.s3;

				sum[1].s6 += rowmem[time].s6*memy0.s2 + rowmem[time].s7*memy0.s3;
				sum[1].s7 += rowmem[time].s7*memy0.s2 + rowmem[time].s6*memy0.s3;

				sum[1].s8 += rowmem[time].s8*memy0.s2 + rowmem[time].s9*memy0.s3;
				sum[1].s9 += rowmem[time].s9*memy0.s2 + rowmem[time].s8*memy0.s3;

				sum[1].sA += rowmem[time].sA*memy0.s2 + rowmem[time].sB*memy0.s3;
				sum[1].sB += rowmem[time].sB*memy0.s2 + rowmem[time].sA*memy0.s3;

				sum[1].sC += rowmem[time].sC*memy0.s2 + rowmem[time].sD*memy0.s3;
				sum[1].sD += rowmem[time].sD*memy0.s2 + rowmem[time].sC*memy0.s3;

				sum[1].sE += rowmem[time].sE*memy0.s2 + rowmem[time].sF*memy0.s3;
				sum[1].sF += rowmem[time].sF*memy0.s2 + rowmem[time].sE*memy0.s3;

				sum[2].s0 += rowmem[time].s0*memy0.s4 + rowmem[time].s1*memy0.s5;
				sum[2].s1 += rowmem[time].s1*memy0.s4 + rowmem[time].s0*memy0.s5;

				sum[2].s2 += rowmem[time].s2*memy0.s4 + rowmem[time].s3*memy0.s5;
				sum[2].s3 += rowmem[time].s3*memy0.s4 + rowmem[time].s2*memy0.s5;

				sum[2].s4 += rowmem[time].s4*memy0.s4 + rowmem[time].s5*memy0.s5;
				sum[2].s5 += rowmem[time].s5*memy0.s4 + rowmem[time].s4*memy0.s5;

				sum[2].s6 += rowmem[time].s6*memy0.s4 + rowmem[time].s7*memy0.s5;
				sum[2].s7 += rowmem[time].s7*memy0.s4 + rowmem[time].s6*memy0.s5;

				sum[2].s8 += rowmem[time].s8*memy0.s4 + rowmem[time].s9*memy0.s5;
				sum[2].s9 += rowmem[time].s9*memy0.s4 + rowmem[time].s8*memy0.s5;

				sum[2].sA += rowmem[time].sA*memy0.s4 + rowmem[time].sB*memy0.s5;
				sum[2].sB += rowmem[time].sB*memy0.s4 + rowmem[time].sA*memy0.s5;

				sum[2].sC += rowmem[time].sC*memy0.s4 + rowmem[time].sD*memy0.s5;
				sum[2].sD += rowmem[time].sD*memy0.s4 + rowmem[time].sC*memy0.s5;

				sum[2].sE += rowmem[time].sE*memy0.s4 + rowmem[time].sF*memy0.s5;
				sum[2].sF += rowmem[time].sF*memy0.s4 + rowmem[time].sE*memy0.s5;

				sum[3].s0 += rowmem[time].s0*memy0.s6 + rowmem[time].s1*memy0.s7;
				sum[3].s1 += rowmem[time].s1*memy0.s6 + rowmem[time].s0*memy0.s7;

				sum[3].s2 += rowmem[time].s2*memy0.s6 + rowmem[time].s3*memy0.s7;
				sum[3].s3 += rowmem[time].s3*memy0.s6 + rowmem[time].s2*memy0.s7;

				sum[3].s4 += rowmem[time].s4*memy0.s6 + rowmem[time].s5*memy0.s7;
				sum[3].s5 += rowmem[time].s5*memy0.s6 + rowmem[time].s4*memy0.s7;

				sum[3].s6 += rowmem[time].s6*memy0.s6 + rowmem[time].s7*memy0.s7;
				sum[3].s7 += rowmem[time].s7*memy0.s6 + rowmem[time].s6*memy0.s7;

				sum[3].s8 += rowmem[time].s8*memy0.s6 + rowmem[time].s9*memy0.s7;
				sum[3].s9 += rowmem[time].s9*memy0.s6 + rowmem[time].s8*memy0.s7;

				sum[3].sA += rowmem[time].sA*memy0.s6 + rowmem[time].sB*memy0.s7;
				sum[3].sB += rowmem[time].sB*memy0.s6 + rowmem[time].sA*memy0.s7;

				sum[3].sC += rowmem[time].sC*memy0.s6 + rowmem[time].sD*memy0.s7;
				sum[3].sD += rowmem[time].sD*memy0.s6 + rowmem[time].sC*memy0.s7;

				sum[3].sE += rowmem[time].sE*memy0.s6 + rowmem[time].sF*memy0.s7;
				sum[3].sF += rowmem[time].sF*memy0.s6 + rowmem[time].sE*memy0.s7;

				sum[4].s0 += rowmem[time].s0*memy0.s8 + rowmem[time].s1*memy0.s9;
				sum[4].s1 += rowmem[time].s1*memy0.s8 + rowmem[time].s0*memy0.s9;

				sum[4].s2 += rowmem[time].s2*memy0.s8 + rowmem[time].s3*memy0.s9;
				sum[4].s3 += rowmem[time].s3*memy0.s8 + rowmem[time].s2*memy0.s9;

				sum[4].s4 += rowmem[time].s4*memy0.s8 + rowmem[time].s5*memy0.s9;
				sum[4].s5 += rowmem[time].s5*memy0.s8 + rowmem[time].s4*memy0.s9;

				sum[4].s6 += rowmem[time].s6*memy0.s8 + rowmem[time].s7*memy0.s9;
				sum[4].s7 += rowmem[time].s7*memy0.s8 + rowmem[time].s6*memy0.s9;

				sum[4].s8 += rowmem[time].s8*memy0.s8 + rowmem[time].s9*memy0.s9;
				sum[4].s9 += rowmem[time].s9*memy0.s8 + rowmem[time].s8*memy0.s9;

				sum[4].sA += rowmem[time].sA*memy0.s8 + rowmem[time].sB*memy0.s9;
				sum[4].sB += rowmem[time].sB*memy0.s8 + rowmem[time].sA*memy0.s9;

				sum[4].sC += rowmem[time].sC*memy0.s8 + rowmem[time].sD*memy0.s9;
				sum[4].sD += rowmem[time].sD*memy0.s8 + rowmem[time].sC*memy0.s9;

				sum[4].sE += rowmem[time].sE*memy0.s8 + rowmem[time].sF*memy0.s9;
				sum[4].sF += rowmem[time].sF*memy0.s8 + rowmem[time].sE*memy0.s9;

				sum[5].s0 += rowmem[time].s0*memy0.sA + rowmem[time].s1*memy0.sB;
				sum[5].s1 += rowmem[time].s1*memy0.sA + rowmem[time].s0*memy0.sB;

				sum[5].s2 += rowmem[time].s2*memy0.sA + rowmem[time].s3*memy0.sB;
				sum[5].s3 += rowmem[time].s3*memy0.sA + rowmem[time].s2*memy0.sB;

				sum[5].s4 += rowmem[time].s4*memy0.sA + rowmem[time].s5*memy0.sB;
				sum[5].s5 += rowmem[time].s5*memy0.sA + rowmem[time].s4*memy0.sB;

				sum[5].s6 += rowmem[time].s6*memy0.sA + rowmem[time].s7*memy0.sB;
				sum[5].s7 += rowmem[time].s7*memy0.sA + rowmem[time].s6*memy0.sB;

				sum[5].s8 += rowmem[time].s8*memy0.sA + rowmem[time].s9*memy0.sB;
				sum[5].s9 += rowmem[time].s9*memy0.sA + rowmem[time].s8*memy0.sB;

				sum[5].sA += rowmem[time].sA*memy0.sA + rowmem[time].sB*memy0.sB;
				sum[5].sB += rowmem[time].sB*memy0.sA + rowmem[time].sA*memy0.sB;

				sum[5].sC += rowmem[time].sC*memy0.sA + rowmem[time].sD*memy0.sB;
				sum[5].sD += rowmem[time].sD*memy0.sA + rowmem[time].sC*memy0.sB;

				sum[5].sE += rowmem[time].sE*memy0.sA + rowmem[time].sF*memy0.sB;
				sum[5].sF += rowmem[time].sF*memy0.sA + rowmem[time].sE*memy0.sB;

				sum[6].s0 += rowmem[time].s0*memy0.sC + rowmem[time].s1*memy0.sD;
				sum[6].s1 += rowmem[time].s1*memy0.sC + rowmem[time].s0*memy0.sD;

				sum[6].s2 += rowmem[time].s2*memy0.sC + rowmem[time].s3*memy0.sD;
				sum[6].s3 += rowmem[time].s3*memy0.sC + rowmem[time].s2*memy0.sD;

				sum[6].s4 += rowmem[time].s4*memy0.sC + rowmem[time].s5*memy0.sD;
				sum[6].s5 += rowmem[time].s5*memy0.sC + rowmem[time].s4*memy0.sD;

				sum[6].s6 += rowmem[time].s6*memy0.sC + rowmem[time].s7*memy0.sD;
				sum[6].s7 += rowmem[time].s7*memy0.sC + rowmem[time].s6*memy0.sD;

				sum[6].s8 += rowmem[time].s8*memy0.sC + rowmem[time].s9*memy0.sD;
				sum[6].s9 += rowmem[time].s9*memy0.sC + rowmem[time].s8*memy0.sD;

				sum[6].sA += rowmem[time].sA*memy0.sC + rowmem[time].sB*memy0.sD;
				sum[6].sB += rowmem[time].sB*memy0.sC + rowmem[time].sA*memy0.sD;

				sum[6].sC += rowmem[time].sC*memy0.sC + rowmem[time].sD*memy0.sD;
				sum[6].sD += rowmem[time].sD*memy0.sC + rowmem[time].sC*memy0.sD;

				sum[6].sE += rowmem[time].sE*memy0.sC + rowmem[time].sF*memy0.sD;
				sum[6].sF += rowmem[time].sF*memy0.sC + rowmem[time].sE*memy0.sD;

				sum[7].s0 += rowmem[time].s0*memy0.sE + rowmem[time].s1*memy0.sF;
				sum[7].s1 += rowmem[time].s1*memy0.sE + rowmem[time].s0*memy0.sF;

				sum[7].s2 += rowmem[time].s2*memy0.sE + rowmem[time].s3*memy0.sF;
				sum[7].s3 += rowmem[time].s3*memy0.sE + rowmem[time].s2*memy0.sF;

				sum[7].s4 += rowmem[time].s4*memy0.sE + rowmem[time].s5*memy0.sF;
				sum[7].s5 += rowmem[time].s5*memy0.sE + rowmem[time].s4*memy0.sF;

				sum[7].s6 += rowmem[time].s6*memy0.sE + rowmem[time].s7*memy0.sF;
				sum[7].s7 += rowmem[time].s7*memy0.sE + rowmem[time].s6*memy0.sF;

				sum[7].s8 += rowmem[time].s8*memy0.sE + rowmem[time].s9*memy0.sF;
				sum[7].s9 += rowmem[time].s9*memy0.sE + rowmem[time].s8*memy0.sF;

				sum[7].sA += rowmem[time].sA*memy0.sE + rowmem[time].sB*memy0.sF;
				sum[7].sB += rowmem[time].sB*memy0.sE + rowmem[time].sA*memy0.sF;

				sum[7].sC += rowmem[time].sC*memy0.sE + rowmem[time].sD*memy0.sF;
				sum[7].sD += rowmem[time].sD*memy0.sE + rowmem[time].sC*memy0.sF;

				sum[7].sE += rowmem[time].sE*memy0.sE + rowmem[time].sF*memy0.sF;
				sum[7].sF += rowmem[time].sF*memy0.sE + rowmem[time].sE*memy0.sF;

				sum[8].s0 += rowmem[time].s0*memy1.s0 + rowmem[time].s1*memy1.s1;
				sum[8].s1 += rowmem[time].s1*memy1.s0 + rowmem[time].s0*memy1.s1;

				sum[8].s2 += rowmem[time].s2*memy1.s0 + rowmem[time].s3*memy1.s1;
				sum[8].s3 += rowmem[time].s3*memy1.s0 + rowmem[time].s2*memy1.s1;

				sum[8].s4 += rowmem[time].s4*memy1.s0 + rowmem[time].s5*memy1.s1;
				sum[8].s5 += rowmem[time].s5*memy1.s0 + rowmem[time].s4*memy1.s1;

				sum[8].s6 += rowmem[time].s6*memy1.s0 + rowmem[time].s7*memy1.s1;
				sum[8].s7 += rowmem[time].s7*memy1.s0 + rowmem[time].s6*memy1.s1;

				sum[8].s8 += rowmem[time].s8*memy1.s0 + rowmem[time].s9*memy1.s1;
				sum[8].s9 += rowmem[time].s9*memy1.s0 + rowmem[time].s8*memy1.s1;

				sum[8].sA += rowmem[time].sA*memy1.s0 + rowmem[time].sB*memy1.s1;
				sum[8].sB += rowmem[time].sB*memy1.s0 + rowmem[time].sA*memy1.s1;

				sum[8].sC += rowmem[time].sC*memy1.s0 + rowmem[time].sD*memy1.s1;
				sum[8].sD += rowmem[time].sD*memy1.s0 + rowmem[time].sC*memy1.s1;

				sum[8].sE += rowmem[time].sE*memy1.s0 + rowmem[time].sF*memy1.s1;
				sum[8].sF += rowmem[time].sF*memy1.s0 + rowmem[time].sE*memy1.s1;

				sum[9].s0 += rowmem[time].s0*memy1.s2 + rowmem[time].s1*memy1.s3;
				sum[9].s1 += rowmem[time].s1*memy1.s2 + rowmem[time].s0*memy1.s3;

				sum[9].s2 += rowmem[time].s2*memy1.s2 + rowmem[time].s3*memy1.s3;
				sum[9].s3 += rowmem[time].s3*memy1.s2 + rowmem[time].s2*memy1.s3;

				sum[9].s4 += rowmem[time].s4*memy1.s2 + rowmem[time].s5*memy1.s3;
				sum[9].s5 += rowmem[time].s5*memy1.s2 + rowmem[time].s4*memy1.s3;

				sum[9].s6 += rowmem[time].s6*memy1.s2 + rowmem[time].s7*memy1.s3;
				sum[9].s7 += rowmem[time].s7*memy1.s2 + rowmem[time].s6*memy1.s3;

				sum[9].s8 += rowmem[time].s8*memy1.s2 + rowmem[time].s9*memy1.s3;
				sum[9].s9 += rowmem[time].s9*memy1.s2 + rowmem[time].s8*memy1.s3;

				sum[9].sA += rowmem[time].sA*memy1.s2 + rowmem[time].sB*memy1.s3;
				sum[9].sB += rowmem[time].sB*memy1.s2 + rowmem[time].sA*memy1.s3;

				sum[9].sC += rowmem[time].sC*memy1.s2 + rowmem[time].sD*memy1.s3;
				sum[9].sD += rowmem[time].sD*memy1.s2 + rowmem[time].sC*memy1.s3;

				sum[9].sE += rowmem[time].sE*memy1.s2 + rowmem[time].sF*memy1.s3;
				sum[9].sF += rowmem[time].sF*memy1.s2 + rowmem[time].sE*memy1.s3;

				sum[10].s0 += rowmem[time].s0*memy1.s4 + rowmem[time].s1*memy1.s5;
				sum[10].s1 += rowmem[time].s1*memy1.s4 + rowmem[time].s0*memy1.s5;

				sum[10].s2 += rowmem[time].s2*memy1.s4 + rowmem[time].s3*memy1.s5;
				sum[10].s3 += rowmem[time].s3*memy1.s4 + rowmem[time].s2*memy1.s5;

				sum[10].s4 += rowmem[time].s4*memy1.s4 + rowmem[time].s5*memy1.s5;
				sum[10].s5 += rowmem[time].s5*memy1.s4 + rowmem[time].s4*memy1.s5;

				sum[10].s6 += rowmem[time].s6*memy1.s4 + rowmem[time].s7*memy1.s5;
				sum[10].s7 += rowmem[time].s7*memy1.s4 + rowmem[time].s6*memy1.s5;

				sum[10].s8 += rowmem[time].s8*memy1.s4 + rowmem[time].s9*memy1.s5;
				sum[10].s9 += rowmem[time].s9*memy1.s4 + rowmem[time].s8*memy1.s5;

				sum[10].sA += rowmem[time].sA*memy1.s4 + rowmem[time].sB*memy1.s5;
				sum[10].sB += rowmem[time].sB*memy1.s4 + rowmem[time].sA*memy1.s5;

				sum[10].sC += rowmem[time].sC*memy1.s4 + rowmem[time].sD*memy1.s5;
				sum[10].sD += rowmem[time].sD*memy1.s4 + rowmem[time].sC*memy1.s5;

				sum[10].sE += rowmem[time].sE*memy1.s4 + rowmem[time].sF*memy1.s5;
				sum[10].sF += rowmem[time].sF*memy1.s4 + rowmem[time].sE*memy1.s5;

				sum[11].s0 += rowmem[time].s0*memy1.s6 + rowmem[time].s1*memy1.s7;
				sum[11].s1 += rowmem[time].s1*memy1.s6 + rowmem[time].s0*memy1.s7;

				sum[11].s2 += rowmem[time].s2*memy1.s6 + rowmem[time].s3*memy1.s7;
				sum[11].s3 += rowmem[time].s3*memy1.s6 + rowmem[time].s2*memy1.s7;

				sum[11].s4 += rowmem[time].s4*memy1.s6 + rowmem[time].s5*memy1.s7;
				sum[11].s5 += rowmem[time].s5*memy1.s6 + rowmem[time].s4*memy1.s7;

				sum[11].s6 += rowmem[time].s6*memy1.s6 + rowmem[time].s7*memy1.s7;
				sum[11].s7 += rowmem[time].s7*memy1.s6 + rowmem[time].s6*memy1.s7;

				sum[11].s8 += rowmem[time].s8*memy1.s6 + rowmem[time].s9*memy1.s7;
				sum[11].s9 += rowmem[time].s9*memy1.s6 + rowmem[time].s8*memy1.s7;

				sum[11].sA += rowmem[time].sA*memy1.s6 + rowmem[time].sB*memy1.s7;
				sum[11].sB += rowmem[time].sB*memy1.s6 + rowmem[time].sA*memy1.s7;

				sum[11].sC += rowmem[time].sC*memy1.s6 + rowmem[time].sD*memy1.s7;
				sum[11].sD += rowmem[time].sD*memy1.s6 + rowmem[time].sC*memy1.s7;

				sum[11].sE += rowmem[time].sE*memy1.s6 + rowmem[time].sF*memy1.s7;
				sum[11].sF += rowmem[time].sF*memy1.s6 + rowmem[time].sE*memy1.s7;

				sum[12].s0 += rowmem[time].s0*memy1.s8 + rowmem[time].s1*memy1.s9;
				sum[12].s1 += rowmem[time].s1*memy1.s8 + rowmem[time].s0*memy1.s9;

				sum[12].s2 += rowmem[time].s2*memy1.s8 + rowmem[time].s3*memy1.s9;
				sum[12].s3 += rowmem[time].s3*memy1.s8 + rowmem[time].s2*memy1.s9;

				sum[12].s4 += rowmem[time].s4*memy1.s8 + rowmem[time].s5*memy1.s9;
				sum[12].s5 += rowmem[time].s5*memy1.s8 + rowmem[time].s4*memy1.s9;

				sum[12].s6 += rowmem[time].s6*memy1.s8 + rowmem[time].s7*memy1.s9;
				sum[12].s7 += rowmem[time].s7*memy1.s8 + rowmem[time].s6*memy1.s9;

				sum[12].s8 += rowmem[time].s8*memy1.s8 + rowmem[time].s9*memy1.s9;
				sum[12].s9 += rowmem[time].s9*memy1.s8 + rowmem[time].s8*memy1.s9;

				sum[12].sA += rowmem[time].sA*memy1.s8 + rowmem[time].sB*memy1.s9;
				sum[12].sB += rowmem[time].sB*memy1.s8 + rowmem[time].sA*memy1.s9;

				sum[12].sC += rowmem[time].sC*memy1.s8 + rowmem[time].sD*memy1.s9;
				sum[12].sD += rowmem[time].sD*memy1.s8 + rowmem[time].sC*memy1.s9;

				sum[12].sE += rowmem[time].sE*memy1.s8 + rowmem[time].sF*memy1.s9;
				sum[12].sF += rowmem[time].sF*memy1.s8 + rowmem[time].sE*memy1.s9;

				sum[13].s0 += rowmem[time].s0*memy1.sA + rowmem[time].s1*memy1.sB;
				sum[13].s1 += rowmem[time].s1*memy1.sA + rowmem[time].s0*memy1.sB;

				sum[13].s2 += rowmem[time].s2*memy1.sA + rowmem[time].s3*memy1.sB;
				sum[13].s3 += rowmem[time].s3*memy1.sA + rowmem[time].s2*memy1.sB;

				sum[13].s4 += rowmem[time].s4*memy1.sA + rowmem[time].s5*memy1.sB;
				sum[13].s5 += rowmem[time].s5*memy1.sA + rowmem[time].s4*memy1.sB;

				sum[13].s6 += rowmem[time].s6*memy1.sA + rowmem[time].s7*memy1.sB;
				sum[13].s7 += rowmem[time].s7*memy1.sA + rowmem[time].s6*memy1.sB;

				sum[13].s8 += rowmem[time].s8*memy1.sA + rowmem[time].s9*memy1.sB;
				sum[13].s9 += rowmem[time].s9*memy1.sA + rowmem[time].s8*memy1.sB;

				sum[13].sA += rowmem[time].sA*memy1.sA + rowmem[time].sB*memy1.sB;
				sum[13].sB += rowmem[time].sB*memy1.sA + rowmem[time].sA*memy1.sB;

				sum[13].sC += rowmem[time].sC*memy1.sA + rowmem[time].sD*memy1.sB;
				sum[13].sD += rowmem[time].sD*memy1.sA + rowmem[time].sC*memy1.sB;

				sum[13].sE += rowmem[time].sE*memy1.sA + rowmem[time].sF*memy1.sB;
				sum[13].sF += rowmem[time].sF*memy1.sA + rowmem[time].sE*memy1.sB;

				sum[14].s0 += rowmem[time].s0*memy1.sC + rowmem[time].s1*memy1.sD;
				sum[14].s1 += rowmem[time].s1*memy1.sC + rowmem[time].s0*memy1.sD;

				sum[14].s2 += rowmem[time].s2*memy1.sC + rowmem[time].s3*memy1.sD;
				sum[14].s3 += rowmem[time].s3*memy1.sC + rowmem[time].s2*memy1.sD;

				sum[14].s4 += rowmem[time].s4*memy1.sC + rowmem[time].s5*memy1.sD;
				sum[14].s5 += rowmem[time].s5*memy1.sC + rowmem[time].s4*memy1.sD;

				sum[14].s6 += rowmem[time].s6*memy1.sC + rowmem[time].s7*memy1.sD;
				sum[14].s7 += rowmem[time].s7*memy1.sC + rowmem[time].s6*memy1.sD;

				sum[14].s8 += rowmem[time].s8*memy1.sC + rowmem[time].s9*memy1.sD;
				sum[14].s9 += rowmem[time].s9*memy1.sC + rowmem[time].s8*memy1.sD;

				sum[14].sA += rowmem[time].sA*memy1.sC + rowmem[time].sB*memy1.sD;
				sum[14].sB += rowmem[time].sB*memy1.sC + rowmem[time].sA*memy1.sD;

				sum[14].sC += rowmem[time].sC*memy1.sC + rowmem[time].sD*memy1.sD;
				sum[14].sD += rowmem[time].sD*memy1.sC + rowmem[time].sC*memy1.sD;

				sum[14].sE += rowmem[time].sE*memy1.sC + rowmem[time].sF*memy1.sD;
				sum[14].sF += rowmem[time].sF*memy1.sC + rowmem[time].sE*memy1.sD;

				sum[15].s0 += rowmem[time].s0*memy1.sE + rowmem[time].s1*memy1.sF;
				sum[15].s1 += rowmem[time].s1*memy1.sE + rowmem[time].s0*memy1.sF;

				sum[15].s2 += rowmem[time].s2*memy1.sE + rowmem[time].s3*memy1.sF;
				sum[15].s3 += rowmem[time].s3*memy1.sE + rowmem[time].s2*memy1.sF;

				sum[15].s4 += rowmem[time].s4*memy1.sE + rowmem[time].s5*memy1.sF;
				sum[15].s5 += rowmem[time].s5*memy1.sE + rowmem[time].s4*memy1.sF;

				sum[15].s6 += rowmem[time].s6*memy1.sE + rowmem[time].s7*memy1.sF;
				sum[15].s7 += rowmem[time].s7*memy1.sE + rowmem[time].s6*memy1.sF;

				sum[15].s8 += rowmem[time].s8*memy1.sE + rowmem[time].s9*memy1.sF;
				sum[15].s9 += rowmem[time].s9*memy1.sE + rowmem[time].s8*memy1.sF;

				sum[15].sA += rowmem[time].sA*memy1.sE + rowmem[time].sB*memy1.sF;
				sum[15].sB += rowmem[time].sB*memy1.sE + rowmem[time].sA*memy1.sF;

				sum[15].sC += rowmem[time].sC*memy1.sE + rowmem[time].sD*memy1.sF;
				sum[15].sD += rowmem[time].sD*memy1.sE + rowmem[time].sC*memy1.sF;

				sum[15].sE += rowmem[time].sE*memy1.sE + rowmem[time].sF*memy1.sF;
				sum[15].sF += rowmem[time].sF*memy1.sE + rowmem[time].sE*memy1.sF;


			}
		
			#pragma unroll
			for(int i = 0 ; i < BLOCK_SIZE_Y ; i++){
					(*output)[ch][baselineBlock][i]= sum[i];
			}

			blocky+=2;
			
	  	}	
	}
	  
}


