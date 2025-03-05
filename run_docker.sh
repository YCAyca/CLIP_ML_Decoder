docker run -it --gpus '"device=0"'  \
    -v /home/yca/CLIP_ML_Decoder:/workspace/CLIP_ML_Decoder \
    -v /home/yca/MLDataset:/workspace/MLDataset \
    clip_ml_decoder:1.0 bash
