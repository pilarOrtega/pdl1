#!/bin/bash

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'Daisy' -n 1 -m 'BottomUp' --flag 2 -j -2 -d "1"| tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/Daisy_complete.txt

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_Daisy_level0207.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'Daisy' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_XceptionDAB_level0207.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'XceptionDAB' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_XceptionH_level0207.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'XceptionH' --min 5 --max 50 -s 1
