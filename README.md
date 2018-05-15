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

Để tạo bộ word_mapper, chứa map N từ dùng làm từ điển cho traing. Chạy  
python main.py -create-mapper -csv-folder-path PATH_TO_CSV_FOLDER -vocabulary_size VOCABULARY_SIZE -save-path SAVE_MAPPER_PATH  
* PATH_TO_CSV_FOLDER : Path đến thư mục chứa các file csv để tạo từ điển, vd ./data/*.csv  
* VOCABULARY_SIZE : Số lượng từ trong từ điển, vd 10000 (mặc định 100000)  
* SAVE_MAPPER_PATH : Folder chứa file mapper. vd ./data  

VD:  
python main.py -create-mapper -csv-folder-path ./data/*.csv -vocabulary_size 10000 -save-path ./data  
Output: ./data/word_mapper.json  

Để tạo file config train chứa thông tin training  
python main.py -create-config -csv-folder-path ./data/*.csv -save-path ./data  
Output: ./data/config.json  

Để training, copy file config vào thư mục rỗng, thư mục này để chứa quá trình train. Vd ./train_progress/config.json  
Chạy  
  
python main.py -train -save-path SAVE_PATH -mapper-path MAPPER_PATH  
* SAVE_PATH : Path thư mục chứa file config đã tạo ở trên  
* MAPPER_PATH : Path đến file json của word_mapper.json  
  
VD:  
python main.py -train -save-path "./temp/shortdata" -mapper-path "./temp/word_mapper.json"  
Output: Model chứa thông tin word2vec  

!! Nếu quá trình train bị dừng lại giữa chừng, tiếp tục chạy lại trên save folder đó, chương trình sẽ tự động resume theo file progress.json  

