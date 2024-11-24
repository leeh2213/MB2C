# preprocess ThingsEEG dataset
# We provide python scripts for feature extraction and data preprocessing.
# Take CLIP as the image encoder for example

# 1st step (Extract features from raw images using pretrained CLIP--openai)
python ../dnn_feature_extraction/obtain_feature_maps_clip.py
# 2nd step (PCA and manifold mixup for extracted features from 1st step operation)
python ../dnn_feature_extraction/feature_maps_clip.py

# 3rd step (Extract features from test raw images using pretrained CLIP--openai)
python ../dnn_feature_extraction/feature_maps_clip.py


# preprocess EEGCVPR40 dataset
# filter the data to 5hz-75hz
python ../preprocessing/preprocess_cvpr40.py