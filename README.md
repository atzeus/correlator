Our board has 1,518 DSP blocks

# correlator

Correlator block will not route blocksize > 10 even though Arriana 10 has 1536 DSP blocks, and uses 4 per sum[i][j]+=mulConj(memx[i],memy[j]). Theoretically, we should be able to handle blocks of max 19x19.

The message is:

Error (184036): Cannot place the following 18 DSP cells -- a legal placement which satisfies all the DSP requirements could not be found
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_3_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_2_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_1_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_144_hfp_0_0_|mac_fp_wys_01"
   
The current theory is that while there are enough DSP nodes, the paths on the FPGA become long because of the amount of sharing.

This is prevented by the Trickle architecture which is a systolic array architecture.
The code in Trickle.hs is Haskell code for geneterating code for this.

Results: 


Scatter to memory (writes non-contigous):

Correlate8: 0.0303401 , 178Mhz, 50.16% stall, 49.7% occupancy, 5685MB/s (per bank)
Correlate10: 0.0833752, 151Mhz, 33.28% stall, 13.5% occupancy, 4835MB/s (per bank)
Trickle8: 0.0846193 , 194Mhz 50% stall, 50% occupancy, 6208MB/s (per bank)
Trickle10: 0.0846168 , 148.9Mhz, 30.53% stall, 13.6% occupancy, 4787.5MB/s (per bank) 

Does not really help so far. Apperently this is not the issue.

Trying: write contigous, unroll in time, simplify trickle logic

I do not unstand why the stall is so high. The memory access is contigous and profile says memory access is coalesced, but burst is reported to be 1 instead of 8. Trying non-volatile memory. 

Maybe the point is that we have 2 loads instead of 1 per clock cycle. Resulting in a 50% occupancy.

Trying: a simpler trickle architecture
Fixed precision
Wide reads

# before correlator

A fully unrolled FFT takes the following floor-space:

+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   22%                     ;
; ALUTs                                  ;    8%                     ;
; Dedicated logic registers              ;   13%                     ;
; Memory blocks                          ;   23%                     ;
; DSP blocks                             ;   72%                     ;
+----------------------------------------+---------------------------;

In my experience this is unlikely to route. The FFT constist of 2x a radix8x8fwd call and some multiplications in between. The multiplications in between take up 192 DSP nodes, and the radix8x8fwd takes up 448 DSP nodes (per call).

A fully unrolled 64 channel, 16 taps FIR Filter takes :

+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   32%                     ;
; ALUTs                                  ;   11%                     ;
; Dedicated logic registers              ;   21%                     ;
; Memory blocks                          ;   86%                     ;
; DSP blocks                             ;  135%                     ;
+----------------------------------------+---------------------------;

A single 16 taps fir filter takes up 32 DSP Nodes.


We need some strategy to chop up FIR-filter/FFT in multiple blocks with hardware sharing. FIR Filter is easy to chop up, but FFT requires some thought.
