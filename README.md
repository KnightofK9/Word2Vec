# Word2Vec

Chương trình training word2vec  
Yêu cầu:  
Anaconda cài các package sau:
* jsonpickle  
* tensorflow  
* pandas  
* numpy  
* matplotlib  
* nltk  

  
Thông số:  
* CSV_FOLDER_PATH : Path đến thư mục chứa các file csv để tạo từ điển, vd ./data/*.csv  
* MIN_WORD_COUNT : Số lượng lần xuất hiện tối thiểu để từ có thể dùng trong vocabulary (mặc định sử dụng vocabulary size) 
* SAVE_PATH : Folder chứa file xuất ra. vd ./data   
* WORD_COUNT_PATH : Path đến file word_count.json
* TRAIN_MODEL : cbow hoặc skipgram  (mặc định skipgram)
* TRAIN_MODE : doc2vec hoặc word2vec (mặc định word2vec)  
* TRAIN_TYPE : normal hoặc empty (normal là train bình thường, empty là chỉ xuất kết quả đọc sample label)
* DOC_MAPPER_PATH : Path đến file doc_mapper.json
* WORD_MAPPER_PATH : Path đến file word_mapper.json
* CONFIG_PATH : Path đến file config.json
* VOCABULARY_SIZE : Size của vocabulary (mặc định 10000)  
* SAVE_FOLDER_PATH : Path đến thư mục chứa các file được tạo ra  
* DOC_EMBEDDING_PATH : Path đến doc_embedding.vec  
* MIN_WORD_COUNT : Giá trị mặc định số lần xuất hiện từ để bỏ vào word_mapper (mặc định None, dùng vocabulary size)
-use-preprocessor : Xét khi tạo bộ word_count có sử dụng preprocess hay không.

Name | Command | Example
--- | --- | ---
Create word count| python main.py -create-word-count -csv-folder-path CSV_FOLDER_PATH -save-path SAVE_FOLDER_PATH |  
Create word mapper| python main.py -create-word-mapper -save-path SAVE_FOLDER_PATH [-csv-folder-path CSV_FOLDER_PATH]/[-word-count-path WORD_COUNT_PATH] [-min_word_count MIN_WORD_COUNT]/[-vocabulary-size VOCABULARY_SIZE]| 
Create doc mapper| python main.py -create-doc-mapper -csv-folder-path CSV_FOLDER_PATH -save-path SAVE_FOLDER_PATH |  
Create config| python main.py -create-config -csv-folder-path CSV_FOLDER_PATH -save-path SAVE_FOLDER_PATH -train-model TRAIN_MODEL -train-mode TRAIN_MODE|  
Train word2vec| python main.py -train-type "normal" -config-path CONFIG_PATH -word-mapper-path WORD_MAPPER_PATH|  
Train doc2vec| python main.py -train-type "normal" -config-path CONFIG_PATH -word-mapper-path WORD_MAPPER_PATH -doc-mapper-path DOC_MAPPER_PATH|  
Evaluate doc embedding result| python main.py -eval-doc-embedding -doc-embedding-path DOC_EMBEDDING_PATH -doc-mapper-path DOC_MAPPER_PATH|  
