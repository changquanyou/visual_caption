# data
    data processing and accessing api
## data_config
    configuration for each part of data, such as：
    raw_data, json_data,images,
    caption_txt,vocab, 
    detect,tf_record
    log, checkpoint

## data_loader
    load raw data 
    generate and load embeddings
    
## data_prepare
   prepare necessary medium from raw data, such as
   caption_tokenized_txt
   vocab_txt
   
   defined in data_config
## data_detector
    detect the regions for each image which is loaded by data_loader
    
## data_builder
    build raw data to tfrecord

## data_reader
    read tfrecord data 
    
## data_display
    display raw or generate data

## feature
    visual feature extractor

