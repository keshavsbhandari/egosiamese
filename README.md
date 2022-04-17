There are 10 model training task divided into siamese and non-siamese based training.

All .py files started with train* are different training strategy.
The naming strategy : train_OBJECTIVE_ARCHITECTURE.py

OBJECTIVE: s(spatial), sc(spatial-classifier), m(motion), mc(motion classifier), avgc(average classifier), catc(concatenated classifier)

ARCHITECTURE: siamese, None(non siamese)

# First Part [SIAMESE BASED]

## Step-1:
    Train Siamese Representation
    1. Train Motion Stream : python train_m_siamese.py
    2. Train Spatial Stream: python train_s_siamese.py
    
## Step-2:
    Train classifier
    1. Train Motion Classifier: python train_mc_siamese.py
    2. Train Spatial Classifier: python train_sc_siamese.py
    
## Step-3:
    Train Fused Classifier
    1. Train Avg Classifier : python train_avgc_siamese.py
    2. Train Concatenated Classifier: python train_catc_siamese.py

# First Part [NON SIAMESE BASED]

## Step-1:
    Train classifier
    1. Train Motion Classifier: python train_mc.py
    2. Train Spatial Classifier: python train_sc.py
    
## Step-2:
    Train Fused Classifier
    1. Train Avg Classifier : python train_avgc.py
    2. Train Concatenated Classifier: python train_catc.py
    

