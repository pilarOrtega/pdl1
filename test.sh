#!/bin/bash

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'DenseDAB' -n 1 -m 'BottomUp' -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/DenseDAB_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'DenseH' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/DenseH_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'Dense' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/Dense_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'DaisyDAB' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/DaisyDAB_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'DaisyH' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/DaisyH_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'Daisy' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/Daisy_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'VGG16DAB' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/VGG16DAB_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'VGG16H' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/VGG16H_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'VGG16' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/VGG16_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'XceptionDAB' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/XceptionDAB_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'XceptionH' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/XceptionH_complete.txt

python test.py -s '/home/DeepLearning/media/pdl1/RC.PDL1.V2' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' -l 16 -tr 0.25 -ts 224 -f 'Xception' -n 1 -m 'BottomUp' -f 2 -j -2 | tee /data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/Records/Xception_complete.txt

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_DenseDAB_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'DenseDAB' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_DenseH_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'DenseH' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_Dense_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'Dense' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_DaisyDAB_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'DaisyDAB' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_DaisyH_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'DaisyH' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_Daisy_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'Daisy' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_VGG16DAB_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'VGG16DAB' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_VGG16H_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'VGG16H' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_VGG16_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'VGG16' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_XceptionDAB_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'XceptionDAB' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_XceptionH_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'XceptionH' --min 5 --max 50 -s 1

python test_bestkmeans.py -f '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/features_Xception_level16.p' -c '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207/class_16_224.p' -o '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/Results_0207' --feature_method 'Xception' --min 5 --max 50 -s 1
