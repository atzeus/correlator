#define NR_RECEIVERS		576
#define NR_SAMPLES_PER_CHANNEL	3072
#define CHANNEL_BLOCK 8 
#define NR_CHANNELS		64
#define NR_CHANNEL_BLOCK	(NR_CHANNELS / CHANNEL_BLOCK)
#define NR_TAPS			16

#define	COMPLEX			2

typedef signed char int8_t;
typedef float2 fcomplex;

typedef char2	InputType/*[NR_RECEIVERS]*/[NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1][NR_CHANNELS];
typedef fcomplex  OutputType/*[NR_RECEIVERS]*/[NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];
typedef float WeightsType[NR_CHANNELS][NR_TAPS];
typedef fcomplex vType[NR_CHANNELS];
typedef fcomplex dvType[NR_CHANNELS];

inline float2 cmul(float2 a, float2 b)
{
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}


#define DIR (-1)

typedef float4  float2x2;
typedef float16 float8x2;




inline float8x2 bitreverse8(float8x2 a)
{
  return (float8x2)(a.s018945CD23AB67EF);
}


//inline float2x2 radix2_fwd(float2x2 a)
//{
  //return (float2x2)(a.s01 + a.s23, a.s01 - a.s23);
//}

inline void radix2_fwd(float2 *restrict out0, float2 *restrict out1, float2 in0, float2 in1)
{
  *out0 = in0 + in1;
  *out1 = in0 - in1;
}


inline void radix_8x8_fwd(float2 out[8][8], const float2 in[8][8])
{
  for (int i = 0; i < 8; i ++) {
    float2 b[8], c[8];

    radix2_fwd(&b[0], &b[4], in[i][0], in[i][4]);
    radix2_fwd(&b[1], &b[5], in[i][1], in[i][5]);
    radix2_fwd(&b[2], &b[6], in[i][2], in[i][6]);
    radix2_fwd(&b[3], &b[7], in[i][3], in[i][7]);

    b[5] = cmul(b[5], (float2)(.70710678118654752440f,(DIR)*.70710678118654752440f));
    b[6] = (float2)(DIR*-b[6].y, DIR*b[6].x);
    b[7] = cmul(b[7], (float2)(-.70710678118654752440f,(DIR)*.70710678118654752440f));

    radix2_fwd(&c[0], &c[2], b[0], b[2]);
    radix2_fwd(&c[1], &c[3], b[1], b[3]);
    radix2_fwd(&c[4], &c[6], b[4], b[6]);
    radix2_fwd(&c[5], &c[7], b[5], b[7]);

    c[3] = (float2)(DIR*-c[3].y, DIR*c[3].x);
    c[7] = (float2)(DIR*-c[7].y, DIR*c[7].x);

    radix2_fwd(&out[0][i], &out[4][i], c[0], c[1]);
    radix2_fwd(&out[2][i], &out[6][i], c[2], c[3]);
    radix2_fwd(&out[1][i], &out[5][i], c[4], c[5]);
    radix2_fwd(&out[3][i], &out[7][i], c[6], c[7]);
  }
}


__constant float2 fftweights[8][8] = 
{
  {
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
    (float2)(1.f,0.f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.995184726672f,-0.0980171403296f),
    (float2)(0.980785280403f,-0.195090322016f),
    (float2)(0.956940335732f,-0.290284677254f),
    (float2)(0.923879532511f,-0.382683432365f),
    (float2)(0.881921264348f,-0.471396736826f),
    (float2)(0.831469612303f,-0.55557023302f),
    (float2)(0.773010453363f,-0.634393284164f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.980785280403f,-0.195090322016f),
    (float2)(0.923879532511f,-0.382683432365f),
    (float2)(0.831469612303f,-0.55557023302f),
    (float2)(0.707106781187f,-0.707106781187f),
    (float2)(0.55557023302f,-0.831469612303f),
    (float2)(0.382683432365f,-0.923879532511f),
    (float2)(0.195090322016f,-0.980785280403f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.956940335732f,-0.290284677254f),
    (float2)(0.831469612303f,-0.55557023302f),
    (float2)(0.634393284164f,-0.773010453363f),
    (float2)(0.382683432365f,-0.923879532511f),
    (float2)(0.0980171403296f,-0.995184726672f),
    (float2)(-0.195090322016f,-0.980785280403f),
    (float2)(-0.471396736826f,-0.881921264348f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.923879532511f,-0.382683432365f),
    (float2)(0.707106781187f,-0.707106781187f),
    (float2)(0.382683432365f,-0.923879532511f),
    (float2)(0.f,-1.f),
    (float2)(-0.382683432365f,-0.923879532511f),
    (float2)(-0.707106781187f,-0.707106781187f),
    (float2)(-0.923879532511f,-0.382683432365f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.881921264348f,-0.471396736826f),
    (float2)(0.55557023302f,-0.831469612303f),
    (float2)(0.0980171403296f,-0.995184726672f),
    (float2)(-0.382683432365f,-0.923879532511f),
    (float2)(-0.773010453363f,-0.634393284164f),
    (float2)(-0.980785280403f,-0.195090322016f),
    (float2)(-0.956940335732f,0.290284677254f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.831469612303f,-0.55557023302f),
    (float2)(0.382683432365f,-0.923879532511f),
    (float2)(-0.195090322016f,-0.980785280403f),
    (float2)(-0.707106781187f,-0.707106781187f),
    (float2)(-0.980785280403f,-0.195090322016f),
    (float2)(-0.923879532511f,0.382683432365f),
    (float2)(-0.55557023302f,0.831469612303f),
  },{
    (float2)(1.f,0.f),
    (float2)(0.773010453363f,-0.634393284164f),
    (float2)(0.195090322016f,-0.980785280403f),
    (float2)(-0.471396736826f,-0.881921264348f),
    (float2)(-0.923879532511f,-0.382683432365f),
    (float2)(-0.956940335732f,0.290284677254f),
    (float2)(-0.55557023302f,0.831469612303f),
    (float2)(0.0980171403296f,0.995184726672f),
  }
};

__kernel __attribute__((reqd_work_group_size(1,1,1)))__attribute__((max_global_work_dim(0)))
void FIR_Filter(__global OutputType *restrict output, const __global InputType *restrict input, const __global WeightsType *restrict filterWeights, const __global vType *restrict vi, const __global dvType *restrict dvi)
{

	float weights[NR_CHANNEL_BLOCK][CHANNEL_BLOCK][NR_TAPS];
	char2 history[NR_CHANNEL_BLOCK][CHANNEL_BLOCK][NR_TAPS];
	for(int chb = 0 ; chb < NR_CHANNEL_BLOCK ; chb++){
		for(int ch = 0 ; ch < CHANNEL_BLOCK ; ch++){
			for (int i = 0; i < NR_TAPS; i ++) {
				weights[chb][ch][i] = (*filterWeights)[chb * CHANNEL_BLOCK + ch][i];
			}
		}
	}
	for(int chb = 0 ; chb < NR_CHANNEL_BLOCK ; chb++){
		for(int ch = 0 ; ch < CHANNEL_BLOCK ; ch++){
			for (int i = 0; i < NR_TAPS; i ++) {
				history[chb][ch][i] = (char2)0;
			}
		}
	}
	/*
  float2 v[NR_CHANNEL_BLOCK][CHANNEL_BLOCK], dv[NR_CHANNEL_BLOCK][CHANNEL_BLOCK];
	for(int chb = 0 ; chb < NR_CHANNEL_BLOCK ; chb++){
		for(int ch = 0 ; ch < CHANNEL_BLOCK ; ch++){
			v[chb][ch] = (*vi)[chb * CHANNEL_BLOCK + ch];
			dv[chb][ch] = (*dvi)[chb * CHANNEL_BLOCK + ch];
		}
	}
	(*/
	for (uint time = 0; time < NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1; time ++) {
		fcomplex sum[NR_CHANNEL_BLOCK][CHANNEL_BLOCK];
		#pragma unroll
		for(int chb = 0 ; chb < NR_CHANNEL_BLOCK ; chb++){
			#pragma unroll
			for(int ch = 0 ; ch < CHANNEL_BLOCK ; ch++){
				#pragma unroll
				for(int i = 0; i < NR_TAPS - 1; i++){
					history[chb][ch][i+1] = history[chb][ch][i];
				}
				history[chb][ch][0] = (*input)[time][ch];
				fcomplex s = (fcomplex)0;
				#pragma unroll
				for(int i = 0 ; i < NR_TAPS ; i++){
					s+= (float2)((float)(history[chb][ch][i].s0), (float)(history[chb][ch][i].s1)) * weights[chb][ch][i];
				}
		  (*output)[time - NR_TAPS - 1][chb * NR_CHANNEL_BLOCK + ch]= s;
			}
		}
		/*
		float2  b[8][8], c[8][8], fftout[8][8];

	  	radix_8x8_fwd(b, sum);
	  	#pragma unroll
	  	for (int i = 0; i < 8; i ++) {
	  		#pragma unroll
			for (int j = 0; j < 8; j ++) {
		  	c[i][j] = cmul(b[i][j], fftweights[i][j]);
			}
		}

	  	radix_8x8_fwd(fftout, c);

		unsigned chb = 0;
		unsigned chbm = 0;
		#pragma unroll
	 	 for (unsigned ch = 0; ch < NR_CHANNELS; ch ++) {
			if (time >=  NR_TAPS - 1) {
		  	(*output)[time - NR_TAPS - 1][ch] = cmul(v[chb][chbm], fftout[chb][chbm]);
		  	v[chb][chbm] = cmul(v[chb][chbm], dv[chb][chbm]);
			}
			chbm++;
			if(chbm == CHANNEL_BLOCK){
				chb++;
				chbm = 0;
			}
		}
		*/
	}
}



#if 0
float8x2 radix8_fwd(float8x2 a)
{
  a.s0189 = radix2_fwd(a.s0189);
  a.s23AB = radix2_fwd(a.s23AB);
  a.s45CD = radix2_fwd(a.s45CD);
  a.s67EF = radix2_fwd(a.s67EF);
  a.sAB   = cmul(a.sAB, (float2)(.70710678118654752440f,(DIR)*.70710678118654752440f));
  a.sCD   = (float2)(DIR*-a.sD,DIR*a.sC);
  a.sEF   = cmul(a.sEF, (float2)(-.70710678118654752440f,(DIR)*.70710678118654752440f));

  a.s0145 = radix2_fwd(a.s0145);
  a.s2367 = radix2_fwd(a.s2367);
  a.s89CD = radix2_fwd(a.s89CD);
  a.sABEF = radix2_fwd(a.sABEF);
  a.s67   = (float2)(DIR*-a.s7,DIR*a.s6);
  a.sEF   = (float2)(DIR*-a.sF,DIR*a.sE);

  a.s0123 = radix2_fwd(a.s0123);
  a.s4567 = radix2_fwd(a.s4567);
  a.s89AB = radix2_fwd(a.s89AB);
  a.sCDEF = radix2_fwd(a.sCDEF);

  return bitreverse8(a);
}
#endif





