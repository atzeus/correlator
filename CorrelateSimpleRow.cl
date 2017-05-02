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
typedef fcomplex8 OutputType[NR_CHANNELS][NR_BLOCKS][2][BLOCK_SIZE]; 

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
			
			fcomplex8 sum[2][BLOCK_SIZE];
	 		#pragma unroll
	 		for(int ysub = 0 ; ysub < 2 ; ysub++){
		 		#pragma unroll
				for(int i = 0 ; i < BLOCK_SIZE ; i++){
					sum[ysub][i] = (fcomplex8)0;
				}
			}
			for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL; time++){
				#pragma unroll
				for(int ysub = 0 ; ysub < 2 ; ysub++){
		  			fcomplex8 memy = (*input)[ch][time][blocky + ysub];
					sum[ysub][0].s0 += rowmem[time].s0*memy.s0 + rowmem[time].s1*memy.s1;
					sum[ysub][0].s1 += rowmem[time].s1*memy.s0 + rowmem[time].s0*memy.s1;

					sum[ysub][0].s2 += rowmem[time].s2*memy.s0 + rowmem[time].s3*memy.s1;
					sum[ysub][0].s3 += rowmem[time].s3*memy.s0 + rowmem[time].s2*memy.s1;

					sum[ysub][0].s4 += rowmem[time].s4*memy.s0 + rowmem[time].s5*memy.s1;
					sum[ysub][0].s5 += rowmem[time].s5*memy.s0 + rowmem[time].s4*memy.s1;

					sum[ysub][0].s6 += rowmem[time].s6*memy.s0 + rowmem[time].s7*memy.s1;
					sum[ysub][0].s7 += rowmem[time].s7*memy.s0 + rowmem[time].s6*memy.s1;

					sum[ysub][0].s8 += rowmem[time].s8*memy.s0 + rowmem[time].s9*memy.s1;
					sum[ysub][0].s9 += rowmem[time].s9*memy.s0 + rowmem[time].s8*memy.s1;

					sum[ysub][0].sA += rowmem[time].sA*memy.s0 + rowmem[time].sB*memy.s1;
					sum[ysub][0].sB += rowmem[time].sB*memy.s0 + rowmem[time].sA*memy.s1;

					sum[ysub][0].sC += rowmem[time].sC*memy.s0 + rowmem[time].sD*memy.s1;
					sum[ysub][0].sD += rowmem[time].sD*memy.s0 + rowmem[time].sC*memy.s1;

					sum[ysub][0].sE += rowmem[time].sE*memy.s0 + rowmem[time].sF*memy.s1;
					sum[ysub][0].sF += rowmem[time].sF*memy.s0 + rowmem[time].sE*memy.s1;

					sum[ysub][1].s0 += rowmem[time].s0*memy.s2 + rowmem[time].s1*memy.s3;
					sum[ysub][1].s1 += rowmem[time].s1*memy.s2 + rowmem[time].s0*memy.s3;

					sum[ysub][1].s2 += rowmem[time].s2*memy.s2 + rowmem[time].s3*memy.s3;
					sum[ysub][1].s3 += rowmem[time].s3*memy.s2 + rowmem[time].s2*memy.s3;

					sum[ysub][1].s4 += rowmem[time].s4*memy.s2 + rowmem[time].s5*memy.s3;
					sum[ysub][1].s5 += rowmem[time].s5*memy.s2 + rowmem[time].s4*memy.s3;

					sum[ysub][1].s6 += rowmem[time].s6*memy.s2 + rowmem[time].s7*memy.s3;
					sum[ysub][1].s7 += rowmem[time].s7*memy.s2 + rowmem[time].s6*memy.s3;

					sum[ysub][1].s8 += rowmem[time].s8*memy.s2 + rowmem[time].s9*memy.s3;
					sum[ysub][1].s9 += rowmem[time].s9*memy.s2 + rowmem[time].s8*memy.s3;

					sum[ysub][1].sA += rowmem[time].sA*memy.s2 + rowmem[time].sB*memy.s3;
					sum[ysub][1].sB += rowmem[time].sB*memy.s2 + rowmem[time].sA*memy.s3;

					sum[ysub][1].sC += rowmem[time].sC*memy.s2 + rowmem[time].sD*memy.s3;
					sum[ysub][1].sD += rowmem[time].sD*memy.s2 + rowmem[time].sC*memy.s3;

					sum[ysub][1].sE += rowmem[time].sE*memy.s2 + rowmem[time].sF*memy.s3;
					sum[ysub][1].sF += rowmem[time].sF*memy.s2 + rowmem[time].sE*memy.s3;

					sum[ysub][2].s0 += rowmem[time].s0*memy.s4 + rowmem[time].s1*memy.s5;
					sum[ysub][2].s1 += rowmem[time].s1*memy.s4 + rowmem[time].s0*memy.s5;

					sum[ysub][2].s2 += rowmem[time].s2*memy.s4 + rowmem[time].s3*memy.s5;
					sum[ysub][2].s3 += rowmem[time].s3*memy.s4 + rowmem[time].s2*memy.s5;

					sum[ysub][2].s4 += rowmem[time].s4*memy.s4 + rowmem[time].s5*memy.s5;
					sum[ysub][2].s5 += rowmem[time].s5*memy.s4 + rowmem[time].s4*memy.s5;

					sum[ysub][2].s6 += rowmem[time].s6*memy.s4 + rowmem[time].s7*memy.s5;
					sum[ysub][2].s7 += rowmem[time].s7*memy.s4 + rowmem[time].s6*memy.s5;

					sum[ysub][2].s8 += rowmem[time].s8*memy.s4 + rowmem[time].s9*memy.s5;
					sum[ysub][2].s9 += rowmem[time].s9*memy.s4 + rowmem[time].s8*memy.s5;

					sum[ysub][2].sA += rowmem[time].sA*memy.s4 + rowmem[time].sB*memy.s5;
					sum[ysub][2].sB += rowmem[time].sB*memy.s4 + rowmem[time].sA*memy.s5;

					sum[ysub][2].sC += rowmem[time].sC*memy.s4 + rowmem[time].sD*memy.s5;
					sum[ysub][2].sD += rowmem[time].sD*memy.s4 + rowmem[time].sC*memy.s5;

					sum[ysub][2].sE += rowmem[time].sE*memy.s4 + rowmem[time].sF*memy.s5;
					sum[ysub][2].sF += rowmem[time].sF*memy.s4 + rowmem[time].sE*memy.s5;

					sum[ysub][3].s0 += rowmem[time].s0*memy.s6 + rowmem[time].s1*memy.s7;
					sum[ysub][3].s1 += rowmem[time].s1*memy.s6 + rowmem[time].s0*memy.s7;

					sum[ysub][3].s2 += rowmem[time].s2*memy.s6 + rowmem[time].s3*memy.s7;
					sum[ysub][3].s3 += rowmem[time].s3*memy.s6 + rowmem[time].s2*memy.s7;

					sum[ysub][3].s4 += rowmem[time].s4*memy.s6 + rowmem[time].s5*memy.s7;
					sum[ysub][3].s5 += rowmem[time].s5*memy.s6 + rowmem[time].s4*memy.s7;

					sum[ysub][3].s6 += rowmem[time].s6*memy.s6 + rowmem[time].s7*memy.s7;
					sum[ysub][3].s7 += rowmem[time].s7*memy.s6 + rowmem[time].s6*memy.s7;

					sum[ysub][3].s8 += rowmem[time].s8*memy.s6 + rowmem[time].s9*memy.s7;
					sum[ysub][3].s9 += rowmem[time].s9*memy.s6 + rowmem[time].s8*memy.s7;

					sum[ysub][3].sA += rowmem[time].sA*memy.s6 + rowmem[time].sB*memy.s7;
					sum[ysub][3].sB += rowmem[time].sB*memy.s6 + rowmem[time].sA*memy.s7;

					sum[ysub][3].sC += rowmem[time].sC*memy.s6 + rowmem[time].sD*memy.s7;
					sum[ysub][3].sD += rowmem[time].sD*memy.s6 + rowmem[time].sC*memy.s7;

					sum[ysub][3].sE += rowmem[time].sE*memy.s6 + rowmem[time].sF*memy.s7;
					sum[ysub][3].sF += rowmem[time].sF*memy.s6 + rowmem[time].sE*memy.s7;

					sum[ysub][4].s0 += rowmem[time].s0*memy.s8 + rowmem[time].s1*memy.s9;
					sum[ysub][4].s1 += rowmem[time].s1*memy.s8 + rowmem[time].s0*memy.s9;

					sum[ysub][4].s2 += rowmem[time].s2*memy.s8 + rowmem[time].s3*memy.s9;
					sum[ysub][4].s3 += rowmem[time].s3*memy.s8 + rowmem[time].s2*memy.s9;

					sum[ysub][4].s4 += rowmem[time].s4*memy.s8 + rowmem[time].s5*memy.s9;
					sum[ysub][4].s5 += rowmem[time].s5*memy.s8 + rowmem[time].s4*memy.s9;

					sum[ysub][4].s6 += rowmem[time].s6*memy.s8 + rowmem[time].s7*memy.s9;
					sum[ysub][4].s7 += rowmem[time].s7*memy.s8 + rowmem[time].s6*memy.s9;

					sum[ysub][4].s8 += rowmem[time].s8*memy.s8 + rowmem[time].s9*memy.s9;
					sum[ysub][4].s9 += rowmem[time].s9*memy.s8 + rowmem[time].s8*memy.s9;

					sum[ysub][4].sA += rowmem[time].sA*memy.s8 + rowmem[time].sB*memy.s9;
					sum[ysub][4].sB += rowmem[time].sB*memy.s8 + rowmem[time].sA*memy.s9;

					sum[ysub][4].sC += rowmem[time].sC*memy.s8 + rowmem[time].sD*memy.s9;
					sum[ysub][4].sD += rowmem[time].sD*memy.s8 + rowmem[time].sC*memy.s9;

					sum[ysub][4].sE += rowmem[time].sE*memy.s8 + rowmem[time].sF*memy.s9;
					sum[ysub][4].sF += rowmem[time].sF*memy.s8 + rowmem[time].sE*memy.s9;

					sum[ysub][5].s0 += rowmem[time].s0*memy.sA + rowmem[time].s1*memy.sB;
					sum[ysub][5].s1 += rowmem[time].s1*memy.sA + rowmem[time].s0*memy.sB;

					sum[ysub][5].s2 += rowmem[time].s2*memy.sA + rowmem[time].s3*memy.sB;
					sum[ysub][5].s3 += rowmem[time].s3*memy.sA + rowmem[time].s2*memy.sB;

					sum[ysub][5].s4 += rowmem[time].s4*memy.sA + rowmem[time].s5*memy.sB;
					sum[ysub][5].s5 += rowmem[time].s5*memy.sA + rowmem[time].s4*memy.sB;

					sum[ysub][5].s6 += rowmem[time].s6*memy.sA + rowmem[time].s7*memy.sB;
					sum[ysub][5].s7 += rowmem[time].s7*memy.sA + rowmem[time].s6*memy.sB;

					sum[ysub][5].s8 += rowmem[time].s8*memy.sA + rowmem[time].s9*memy.sB;
					sum[ysub][5].s9 += rowmem[time].s9*memy.sA + rowmem[time].s8*memy.sB;

					sum[ysub][5].sA += rowmem[time].sA*memy.sA + rowmem[time].sB*memy.sB;
					sum[ysub][5].sB += rowmem[time].sB*memy.sA + rowmem[time].sA*memy.sB;

					sum[ysub][5].sC += rowmem[time].sC*memy.sA + rowmem[time].sD*memy.sB;
					sum[ysub][5].sD += rowmem[time].sD*memy.sA + rowmem[time].sC*memy.sB;

					sum[ysub][5].sE += rowmem[time].sE*memy.sA + rowmem[time].sF*memy.sB;
					sum[ysub][5].sF += rowmem[time].sF*memy.sA + rowmem[time].sE*memy.sB;

					sum[ysub][6].s0 += rowmem[time].s0*memy.sC + rowmem[time].s1*memy.sD;
					sum[ysub][6].s1 += rowmem[time].s1*memy.sC + rowmem[time].s0*memy.sD;

					sum[ysub][6].s2 += rowmem[time].s2*memy.sC + rowmem[time].s3*memy.sD;
					sum[ysub][6].s3 += rowmem[time].s3*memy.sC + rowmem[time].s2*memy.sD;

					sum[ysub][6].s4 += rowmem[time].s4*memy.sC + rowmem[time].s5*memy.sD;
					sum[ysub][6].s5 += rowmem[time].s5*memy.sC + rowmem[time].s4*memy.sD;

					sum[ysub][6].s6 += rowmem[time].s6*memy.sC + rowmem[time].s7*memy.sD;
					sum[ysub][6].s7 += rowmem[time].s7*memy.sC + rowmem[time].s6*memy.sD;

					sum[ysub][6].s8 += rowmem[time].s8*memy.sC + rowmem[time].s9*memy.sD;
					sum[ysub][6].s9 += rowmem[time].s9*memy.sC + rowmem[time].s8*memy.sD;

					sum[ysub][6].sA += rowmem[time].sA*memy.sC + rowmem[time].sB*memy.sD;
					sum[ysub][6].sB += rowmem[time].sB*memy.sC + rowmem[time].sA*memy.sD;

					sum[ysub][6].sC += rowmem[time].sC*memy.sC + rowmem[time].sD*memy.sD;
					sum[ysub][6].sD += rowmem[time].sD*memy.sC + rowmem[time].sC*memy.sD;

					sum[ysub][6].sE += rowmem[time].sE*memy.sC + rowmem[time].sF*memy.sD;
					sum[ysub][6].sF += rowmem[time].sF*memy.sC + rowmem[time].sE*memy.sD;

					sum[ysub][7].s0 += rowmem[time].s0*memy.sE + rowmem[time].s1*memy.sF;
					sum[ysub][7].s1 += rowmem[time].s1*memy.sE + rowmem[time].s0*memy.sF;

					sum[ysub][7].s2 += rowmem[time].s2*memy.sE + rowmem[time].s3*memy.sF;
					sum[ysub][7].s3 += rowmem[time].s3*memy.sE + rowmem[time].s2*memy.sF;

					sum[ysub][7].s4 += rowmem[time].s4*memy.sE + rowmem[time].s5*memy.sF;
					sum[ysub][7].s5 += rowmem[time].s5*memy.sE + rowmem[time].s4*memy.sF;

					sum[ysub][7].s6 += rowmem[time].s6*memy.sE + rowmem[time].s7*memy.sF;
					sum[ysub][7].s7 += rowmem[time].s7*memy.sE + rowmem[time].s6*memy.sF;

					sum[ysub][7].s8 += rowmem[time].s8*memy.sE + rowmem[time].s9*memy.sF;
					sum[ysub][7].s9 += rowmem[time].s9*memy.sE + rowmem[time].s8*memy.sF;

					sum[ysub][7].sA += rowmem[time].sA*memy.sE + rowmem[time].sB*memy.sF;
					sum[ysub][7].sB += rowmem[time].sB*memy.sE + rowmem[time].sA*memy.sF;

					sum[ysub][7].sC += rowmem[time].sC*memy.sE + rowmem[time].sD*memy.sF;
					sum[ysub][7].sD += rowmem[time].sD*memy.sE + rowmem[time].sC*memy.sF;

					sum[ysub][7].sE += rowmem[time].sE*memy.sE + rowmem[time].sF*memy.sF;
					sum[ysub][7].sF += rowmem[time].sF*memy.sE + rowmem[time].sE*memy.sF;
				}

			}
		

	 		#pragma unroll
	 		for(int ysub = 0 ; ysub < 2 ; ysub++){
				#pragma unroll
				for(int i = 0 ; i < BLOCK_SIZE; i++){
						(*output)[ch][baselineBlock][ysub][i]= sum[ysub][i];
				}
			}
			blocky+=2;
			
	  	}	
	}
	  
}


