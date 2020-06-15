#!/bin/bash
python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3' -f 'DenseDAB' -m 'BottomUp' --flag 3 -j -2 -c 23| tee Records/test3_DenseDAB_BU_1506-6.txt
python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3' -f 'DaisyDAB' -m 'BottomUp' --flag 2 -j -2 -c 23| tee Records/test3_DaisyDAB_BU_1506-1.txt
