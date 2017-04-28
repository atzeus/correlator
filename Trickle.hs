import Data.List
import System.Environment



initbuf v blocksize = concat [ buftxt i | i <- [1..blocksize-1]]
    where buftxt n = "fcomplex buf" ++ v ++ show n ++ "[" ++ show n ++ "]; \n" ++
                     "#pragma unroll\n" ++ 
                     "for(int i = 0 ; i < " ++ show n ++ "; i++) { \n" ++
                     "  buf" ++ v ++ show n ++ "[i] = (fcomplex)0;\n" ++
                     "}\n"


shiftbuf v blocksize = concat [ buftxt i | i <- [1..blocksize-1 ]]
    where buftxt n = "#pragma unroll\n" ++ 
                     "for(int i = 0 ; i < " ++ show (n - 1) ++ "; i++) { \n" ++
                     "  buf" ++ v ++ show n ++ "[i] = buf" ++ v ++ show n ++ "[i + 1];\n" ++
                     "}\n"
setmems blocksize = "if (time < NR_SAMPLES_PER_CHANNEL) { \n" ++
						(indent ("accum[0][0][0] = (*input)[time][blockx];\n" ++
						         setmemreally "x" blocksize ++ 
						         "accum[0][0][1] = (*input)[time][blocky];\n" ++
						         setmemreally "y" blocksize)) ++
				    "} else { \n" ++
				        indent ("accum[0][0][0] = (fcomplex) 0;\n" ++
						         setmemnaught "x" blocksize ++ 
						         "accum[0][0][1] =  (fcomplex) 0;\n" ++
						         setmemnaught "y" blocksize) ++
			        "}\n"



setmemreally v blocksize = concat [ buftxt i | i <- [1..blocksize -1]]
    where buftxt n = "buf" ++ v ++ show n ++ "[" ++ show (n -1) ++ "] = (*input)[time]" ++ "[block" ++ v ++ " + " ++ show n ++"];\n"

setmemnaught v blocksize = concat [ buftxt i | i <- [1..blocksize -1]]
    where buftxt n = "buf" ++ v ++ show n ++ "[" ++ show (n -1) ++ "] = (fcomplex) 0;\n"

shiftBufToAccumx blocksize =  concat [ buftxt i | i <- [1..blocksize -1]]
    where buftxt n = "accum[0][" ++ show n ++ "][0] = bufx" ++ show n ++ "[0];\n"

shiftBufToAccumy blocksize =   concat [ buftxt i | i <- [1..blocksize -1]]
    where buftxt n = "accum[" ++ show n ++ "][0][1] = bufy" ++ show n ++ "[0];\n"

shiftx  = "#pragma unroll\n" ++ 
                   "for(int i = BLOCK_SIZE -1 ; i > 0;  i--) {\n" ++
                   "  #pragma unroll\n" ++ 
                   "  for(int j = 0 ; j < BLOCK_SIZE; j++) {\n" ++
                   "    accum[i][j][0] = accum[i-1][j][0];\n" ++
                   "  }\n" ++
                   "}\n" 

shifty  = "#pragma unroll\n" ++ 
                   "for(int j = BLOCK_SIZE -1; j > 0;  j--) {\n" ++
                   "  #pragma unroll\n" ++ 
                   "  for(int i = 0 ; i < BLOCK_SIZE; i++) {\n" ++
                   "    accum[i][j][1] = accum[i][j-1][1];\n" ++
                   "  }\n" ++
                   "}\n" 

addAll  =  "#pragma unroll\n" ++ 
                   "for(int i = 0 ; i < BLOCK_SIZE; i++) {\n" ++
                   "  #pragma unroll\n" ++ 
                   "  for(int j = 0 ; j < BLOCK_SIZE ; j++) {\n" ++
                   "    accum[i][j][2] += mulConj(accum[i][j][0], accum[i][j][1]);\n" ++
                   "  }\n" ++
                   "}\n" 

initAccum = intercalate "\n"
            ["fcomplex accum[BLOCK_SIZE][BLOCK_SIZE][3];",
             "#pragma unroll",
             "for(int i = 0 ; i < BLOCK_SIZE  ; i++){",
             "  #pragma unroll",
             "  for(int j = 0 ; j < BLOCK_SIZE ; j++){",
             "    accum[i][j][0] = (fcomplex)0;",
             "    accum[i][j][1] = (fcomplex)0;",
             "    accum[i][j][2] = (fcomplex)0;",
             "  }",
             "}"] ++ "\n"
loadInner = False

getMem = "fcomplex memx[BLOCK_SIZE];\n" ++ 
         "fcomplex memy[BLOCK_SIZE];\n" ++
  if loadInner 
  then ( "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memx[i] =  time < NR_SAMPLES_PER_CHANNEL ? (*input)[time][blockx + i] : (fcomplex)(0);\n" ++
         "}\n" ++
         "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memy[i] =  time < NR_SAMPLES_PER_CHANNEL ? (*input)[time][blocky + i] : (fcomplex)(0);\n" ++
         "}\n" )
  else ("if (time < NR_SAMPLES_PER_CHANNEL) {\n " ++
        indent (
         "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memx[i] =  (*input)[time][blockx + i] ;\n" ++
         "}\n" ++
         "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memy[i] =  (*input)[time][blocky + i];\n" ++
         "}\n") ++ 
         "} else { \n" ++
         indent (
         "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memx[i] =(fcomplex)(0);\n" ++
         "}\n" ++
         "#pragma unroll\n" ++
         "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
         "  memy[i] =(fcomplex)(0);\n" ++
         "}\n" ) ++
         "}\n" 
       )

writemem = "#pragma unroll\n" ++
           "for(int i = 0 ; i < BLOCK_SIZE ; i++){\n" ++
           "  #pragma unroll\n" ++
           "  for(int j = 0 ; j < BLOCK_SIZE ; j++){\n" ++
-- contingous write:
           "  (*output)[baselineBlock][i][j] = accum[i][j][2];\n" ++
           
-- non contigous write:         
{-
          "    int x = blockx + i;\n" ++
           "    int y = blocky + j;\n" ++
           "   int p = x * (x + 1) + y;\n" ++
           "    if( y <= x) { (*output)[p] = accum[i][j][2];}\n" ++
-}
           "  }\n" ++
           "}\n"
           

addIndices = "blocky+=BLOCK_SIZE;\n" ++
             "if(blocky > blockx){\n" ++
             "  blockx+=BLOCK_SIZE;\n" ++
             "  blocky = 0;\n" ++
             "}\n"

indent x = unlines $ map ("  " ++) $ lines x

header blocksize = 
  "#define NR_RECEIVERS  576\n" ++
  "#define NR_BASELINE             166176\n" ++ 
  "#define NR_SAMPLES_PER_CHANNEL 1024\n" ++
  "#define NR_CHANNELS   64\n" ++
  "#define NR_POLARIZATIONS        2\n" ++
  "#define COMPLEX  2\n" ++
  "#define BLOCK_SIZE             " ++ show blocksize ++ "\n" ++
  "#define NR_BLOCK_X             (NR_RECEIVERS / BLOCK_SIZE)\n" ++             
  "#define NR_BLOCKS              (NR_BLOCK_X * (NR_BLOCK_X + 1)) / 2\n" ++
  "#define OUTSIZE                ((NR_RECEIVERS * (NR_RECEIVERS + 1)) / 2)\n" ++
  "\n" ++
  "typedef signed char int8_t;\n" ++      
  "typedef float2 fcomplex;\n" ++      
  "\n" ++
  "typedef fcomplex InputType[NR_SAMPLES_PER_CHANNEL][NR_RECEIVERS]/*[NR_CHANNELS]*/;\n" ++      
-- 
-- non-contigous write:
--  "typedef fcomplex OutputType[OUTSIZE]/*[NR_CHANNELS]*/; \n" ++      
-- contigous write
  "typedef fcomplex OutputType[NR_BLOCKS][BLOCK_SIZE][BLOCK_SIZE]/*[NR_CHANNELS]*/;\n" ++
  "\n" ++
  "fcomplex  __attribute__((__overloadable__,__always_inline__,const)) mulConj(fcomplex a, fcomplex b){\n" ++      
  "  return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);\n" ++      
  "}\n" ++      
  "\n" ++
  "__kernel __attribute__((reqd_work_group_size(1,1,1))) __attribute__((max_global_work_dim(0)))\n" ++      
  "void Correlator(__global OutputType *restrict output, const __global InputType *restrict input)\n" ++      
  "{\n" 

prog blocksize = header blocksize ++ 
         indent ("int blockx = 0;\n" ++
                 "int blocky = 0;\n" ++
                 "for( int baselineBlock = 0 ; baselineBlock < NR_BLOCKS; baselineBlock++) {\n" ++
                  indent (
                    initbuf "x" blocksize ++
                    initbuf "y" blocksize ++
                    initAccum ++
                    "for (int time = 0 ; time < NR_SAMPLES_PER_CHANNEL + BLOCK_SIZE; time++){\n" ++ 
                      indent (
                      shiftx ++ shifty ++ 
                      shiftBufToAccumx blocksize ++ shiftBufToAccumy blocksize ++
                      shiftbuf "x" blocksize ++ shiftbuf "y" blocksize ++
                      setmems blocksize ++
                      addAll) ++
                    "}\n" ++
                    writemem ++ addIndices
                   ) ++ "}\n"
         ) ++ "}\n"
                   
                  
main = do (h:_) <- getArgs
          putStr $ prog (read h)
