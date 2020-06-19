#!/bin/bash
mkdir /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4
python feature_extraction.py -l '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/list_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4' -f 'DenseDAB'

for i in 71 72 73 74 75 76 77 78 79 80
do
  python cluster_division.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4/features_DenseDAB_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4' -m 'BottomUp' --nclusters 23| tee Records/test3_DaisyH_BU_1806-$i.txt
  mv /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class-DenseDAB-16-BottomUp.p /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class-DenseDAB-16-BottomUp-$i.p
done

python feature_extraction.py -l '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/list_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4' -f 'DenseH'

for i in 81 82 83 84 85 86 87 88 89 90
do
  python cluster_division.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4/features_DenseH_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test4' -m 'BottomUp' --nclusters 23| tee Records/test3_DaisyH_BU_1806-$i.txt
  mv /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class-DenseH-16-BottomUp.p /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_test3/class-DenseH-16-BottomUp-$i.p
done
