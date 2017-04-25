# correlator

Correlator block will not route blocksize > 8 even though Arriana 10 has 1536 DSP blocks, and uses 4 per sum[i][j]+=mulConj(memx[i],memy[j]). Theoretically, we should be able to handle blocks of max 19x19.

The message is:

Error (184036): Cannot place the following 18 DSP cells -- a legal placement which satisfies all the DSP requirements could not be found
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_3_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_2_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_1_1_hfp_0_0_|mac_fp_wys_01"
    Info (184037): Node "freeze_wrapper_inst|kernel_system_inst|correlate_block10_system|Correlator|kernel|Correlator_function_inst0|Correlator_basic_block_2|fp_module_local_bb2_sub_144_hfp_0_0_|mac_fp_wys_01"
   
The current theory is that while there are enough DSP nodes, the paths on the FPGA become long because of the amount of sharing.

This is prevented by the Trickle architecture which is a systolic array architecture.
The code in Trickle.hs is Haskell code for geneterating code for this.
