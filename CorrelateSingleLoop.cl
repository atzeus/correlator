#define NR_RECEIVERS		576
#define NR_BASELINE             166176 
#define NR_SAMPLES_PER_CHANNEL	1024
#define NR_CHANNELS		64
#define NR_POLARIZATIONS        2
#define	COMPLEX			2
#define BLOCK_SIZE                 8
#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE)             
#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2
#define OUTSIZE                NR_BLOCKS * BLOCK_SIZE
#define ITERATIONS	(NR_CHANNELS * NR_BLOCKS * NR_SAMPLES_PER_CHANNEL)

typedef signed char int8_t;
typedef float2 fcomplex;
typedef float16 fcomplex8;

typedef fcomplex8 InputType[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS/8];
typedef fcomplex8 OutputType[NR_CHANNELS][NR_BLOCKS][BLOCK_SIZE]; 

/*
fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){
  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}
*/

__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))
void Correlator(__global OutputType *restrict output, const __global volatile InputType *restrict input)
{
	int blockx = 0;
	int blocky = 0;
	int baselineBlock = 0 
	fcomplex8 sum[BLOCK_SIZE];
	int ch = 0; 
	int time = 0;
	for(int nrIteration = 0 ; nrIteration < ITERATIONS; nrIteration++){
			if( time == 0){
		 		#pragma unroll
				for(int i = 0 ; i < BLOCK_SIZE ; i++){
					sum[i] = (fcomplex8)0;
				}
			}
		  fcomplex8 memx = (*input)[ch][time][blockx];
		  fcomplex8 memy = (*input)[ch][time][blocky];
			sum[0].s0 += memx.s0*memy.s0 + memx.s1*memy.s1;
			sum[0].s1 += memx.s1*memy.s0 - memx.s0*memy.s1;

			sum[0].s2 += memx.s2*memy.s0 + memx.s3*memy.s1;
			sum[0].s3 += memx.s3*memy.s0 - memx.s2*memy.s1;

			sum[0].s4 += memx.s4*memy.s0 + memx.s5*memy.s1;
			sum[0].s5 += memx.s5*memy.s0 - memx.s4*memy.s1;

			sum[0].s6 += memx.s6*memy.s0 + memx.s7*memy.s1;
			sum[0].s7 += memx.s7*memy.s0 - memx.s6*memy.s1;

			sum[0].s8 += memx.s8*memy.s0 + memx.s9*memy.s1;
			sum[0].s9 += memx.s9*memy.s0 - memx.s8*memy.s1;

			sum[0].sA += memx.sA*memy.s0 + memx.sB*memy.s1;
			sum[0].sB += memx.sB*memy.s0 - memx.sA*memy.s1;

			sum[0].sC += memx.sC*memy.s0 + memx.sD*memy.s1;
			sum[0].sD += memx.sD*memy.s0 - memx.sC*memy.s1;

			sum[0].sE += memx.sE*memy.s0 + memx.sF*memy.s1;
			sum[0].sF += memx.sF*memy.s0 - memx.sE*memy.s1;

			sum[1].s0 += memx.s0*memy.s2 + memx.s1*memy.s3;
			sum[1].s1 += memx.s1*memy.s2 - memx.s0*memy.s3;

			sum[1].s2 += memx.s2*memy.s2 + memx.s3*memy.s3;
			sum[1].s3 += memx.s3*memy.s2 - memx.s2*memy.s3;

			sum[1].s4 += memx.s4*memy.s2 + memx.s5*memy.s3;
			sum[1].s5 += memx.s5*memy.s2 - memx.s4*memy.s3;

			sum[1].s6 += memx.s6*memy.s2 + memx.s7*memy.s3;
			sum[1].s7 += memx.s7*memy.s2 - memx.s6*memy.s3;

			sum[1].s8 += memx.s8*memy.s2 + memx.s9*memy.s3;
			sum[1].s9 += memx.s9*memy.s2 - memx.s8*memy.s3;

			sum[1].sA += memx.sA*memy.s2 + memx.sB*memy.s3;
			sum[1].sB += memx.sB*memy.s2 - memx.sA*memy.s3;

			sum[1].sC += memx.sC*memy.s2 + memx.sD*memy.s3;
			sum[1].sD += memx.sD*memy.s2 - memx.sC*memy.s3;

			sum[1].sE += memx.sE*memy.s2 + memx.sF*memy.s3;
			sum[1].sF += memx.sF*memy.s2 - memx.sE*memy.s3;

			sum[2].s0 += memx.s0*memy.s4 + memx.s1*memy.s5;
			sum[2].s1 += memx.s1*memy.s4 - memx.s0*memy.s5;

			sum[2].s2 += memx.s2*memy.s4 + memx.s3*memy.s5;
			sum[2].s3 += memx.s3*memy.s4 - memx.s2*memy.s5;

			sum[2].s4 += memx.s4*memy.s4 + memx.s5*memy.s5;
			sum[2].s5 += memx.s5*memy.s4 - memx.s4*memy.s5;

			sum[2].s6 += memx.s6*memy.s4 + memx.s7*memy.s5;
			sum[2].s7 += memx.s7*memy.s4 - memx.s6*memy.s5;

			sum[2].s8 += memx.s8*memy.s4 + memx.s9*memy.s5;
			sum[2].s9 += memx.s9*memy.s4 - memx.s8*memy.s5;

			sum[2].sA += memx.sA*memy.s4 + memx.sB*memy.s5;
			sum[2].sB += memx.sB*memy.s4 - memx.sA*memy.s5;

			sum[2].sC += memx.sC*memy.s4 + memx.sD*memy.s5;
			sum[2].sD += memx.sD*memy.s4 - memx.sC*memy.s5;

			sum[2].sE += memx.sE*memy.s4 + memx.sF*memy.s5;
			sum[2].sF += memx.sF*memy.s4 - memx.sE*memy.s5;

			sum[3].s0 += memx.s0*memy.s6 + memx.s1*memy.s7;
			sum[3].s1 += memx.s1*memy.s6 - memx.s0*memy.s7;

			sum[3].s2 += memx.s2*memy.s6 + memx.s3*memy.s7;
			sum[3].s3 += memx.s3*memy.s6 - memx.s2*memy.s7;

			sum[3].s4 += memx.s4*memy.s6 + memx.s5*memy.s7;
			sum[3].s5 += memx.s5*memy.s6 - memx.s4*memy.s7;

			sum[3].s6 += memx.s6*memy.s6 + memx.s7*memy.s7;
			sum[3].s7 += memx.s7*memy.s6 - memx.s6*memy.s7;

			sum[3].s8 += memx.s8*memy.s6 + memx.s9*memy.s7;
			sum[3].s9 += memx.s9*memy.s6 - memx.s8*memy.s7;

			sum[3].sA += memx.sA*memy.s6 + memx.sB*memy.s7;
			sum[3].sB += memx.sB*memy.s6 - memx.sA*memy.s7;

			sum[3].sC += memx.sC*memy.s6 + memx.sD*memy.s7;
			sum[3].sD += memx.sD*memy.s6 - memx.sC*memy.s7;

			sum[3].sE += memx.sE*memy.s6 + memx.sF*memy.s7;
			sum[3].sF += memx.sF*memy.s6 - memx.sE*memy.s7;

			sum[4].s0 += memx.s0*memy.s8 + memx.s1*memy.s9;
			sum[4].s1 += memx.s1*memy.s8 - memx.s0*memy.s9;

			sum[4].s2 += memx.s2*memy.s8 + memx.s3*memy.s9;
			sum[4].s3 += memx.s3*memy.s8 - memx.s2*memy.s9;

			sum[4].s4 += memx.s4*memy.s8 + memx.s5*memy.s9;
			sum[4].s5 += memx.s5*memy.s8 - memx.s4*memy.s9;

			sum[4].s6 += memx.s6*memy.s8 + memx.s7*memy.s9;
			sum[4].s7 += memx.s7*memy.s8 - memx.s6*memy.s9;

			sum[4].s8 += memx.s8*memy.s8 + memx.s9*memy.s9;
			sum[4].s9 += memx.s9*memy.s8 - memx.s8*memy.s9;

			sum[4].sA += memx.sA*memy.s8 + memx.sB*memy.s9;
			sum[4].sB += memx.sB*memy.s8 - memx.sA*memy.s9;

			sum[4].sC += memx.sC*memy.s8 + memx.sD*memy.s9;
			sum[4].sD += memx.sD*memy.s8 - memx.sC*memy.s9;

			sum[4].sE += memx.sE*memy.s8 + memx.sF*memy.s9;
			sum[4].sF += memx.sF*memy.s8 - memx.sE*memy.s9;

			sum[5].s0 += memx.s0*memy.sA + memx.s1*memy.sB;
			sum[5].s1 += memx.s1*memy.sA - memx.s0*memy.sB;

			sum[5].s2 += memx.s2*memy.sA + memx.s3*memy.sB;
			sum[5].s3 += memx.s3*memy.sA - memx.s2*memy.sB;

			sum[5].s4 += memx.s4*memy.sA + memx.s5*memy.sB;
			sum[5].s5 += memx.s5*memy.sA - memx.s4*memy.sB;

			sum[5].s6 += memx.s6*memy.sA + memx.s7*memy.sB;
			sum[5].s7 += memx.s7*memy.sA - memx.s6*memy.sB;

			sum[5].s8 += memx.s8*memy.sA + memx.s9*memy.sB;
			sum[5].s9 += memx.s9*memy.sA - memx.s8*memy.sB;

			sum[5].sA += memx.sA*memy.sA + memx.sB*memy.sB;
			sum[5].sB += memx.sB*memy.sA - memx.sA*memy.sB;

			sum[5].sC += memx.sC*memy.sA + memx.sD*memy.sB;
			sum[5].sD += memx.sD*memy.sA - memx.sC*memy.sB;

			sum[5].sE += memx.sE*memy.sA + memx.sF*memy.sB;
			sum[5].sF += memx.sF*memy.sA - memx.sE*memy.sB;

			sum[6].s0 += memx.s0*memy.sC + memx.s1*memy.sD;
			sum[6].s1 += memx.s1*memy.sC - memx.s0*memy.sD;

			sum[6].s2 += memx.s2*memy.sC + memx.s3*memy.sD;
			sum[6].s3 += memx.s3*memy.sC - memx.s2*memy.sD;

			sum[6].s4 += memx.s4*memy.sC + memx.s5*memy.sD;
			sum[6].s5 += memx.s5*memy.sC - memx.s4*memy.sD;

			sum[6].s6 += memx.s6*memy.sC + memx.s7*memy.sD;
			sum[6].s7 += memx.s7*memy.sC - memx.s6*memy.sD;

			sum[6].s8 += memx.s8*memy.sC + memx.s9*memy.sD;
			sum[6].s9 += memx.s9*memy.sC - memx.s8*memy.sD;

			sum[6].sA += memx.sA*memy.sC + memx.sB*memy.sD;
			sum[6].sB += memx.sB*memy.sC - memx.sA*memy.sD;

			sum[6].sC += memx.sC*memy.sC + memx.sD*memy.sD;
			sum[6].sD += memx.sD*memy.sC - memx.sC*memy.sD;

			sum[6].sE += memx.sE*memy.sC + memx.sF*memy.sD;
			sum[6].sF += memx.sF*memy.sC - memx.sE*memy.sD;

			sum[7].s0 += memx.s0*memy.sE + memx.s1*memy.sF;
			sum[7].s1 += memx.s1*memy.sE - memx.s0*memy.sF;

			sum[7].s2 += memx.s2*memy.sE + memx.s3*memy.sF;
			sum[7].s3 += memx.s3*memy.sE - memx.s2*memy.sF;

			sum[7].s4 += memx.s4*memy.sE + memx.s5*memy.sF;
			sum[7].s5 += memx.s5*memy.sE - memx.s4*memy.sF;

			sum[7].s6 += memx.s6*memy.sE + memx.s7*memy.sF;
			sum[7].s7 += memx.s7*memy.sE - memx.s6*memy.sF;

			sum[7].s8 += memx.s8*memy.sE + memx.s9*memy.sF;
			sum[7].s9 += memx.s9*memy.sE - memx.s8*memy.sF;

			sum[7].sA += memx.sA*memy.sE + memx.sB*memy.sF;
			sum[7].sB += memx.sB*memy.sE - memx.sA*memy.sF;

			sum[7].sC += memx.sC*memy.sE + memx.sD*memy.sF;
			sum[7].sD += memx.sD*memy.sE - memx.sC*memy.sF;

			sum[7].sE += memx.sE*memy.sE + memx.sF*memy.sF;
			sum[7].sF += memx.sF*memy.sE - memx.sE*memy.sF;

			if( time == NR_SAMPLES_PER_CHANNEL - 1){
		 		#pragma unroll
				for(int i = 0 ; i < BLOCK_SIZE ; i++){
					(*output)[ch][baselineBlock][i]= sum[i];
				}
				time = 0;
				if(baselineBlock == NR_BLOCKS - 1){					
					baselineBlock = 0;
					blockx=0;
					blocky=0;
					ch++;
				} else {
					baselineBlock++;
					if(blocky == blockx){
						blockx+=1;
						blocky = 0;
					} else {
						blocky+=1;
					}
				}
		} else{
			time++;
		}
}


