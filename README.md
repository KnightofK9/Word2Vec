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
* PATH_TO_CSV_FOLDER : Path đến thư mục chứa các file csv để tạo từ điển, vd ./data/*.csv  
* VOCABULARY_SIZE : Số lượng từ trong từ điển, vd 10000 (mặc định 100000)  
* SAVE_PATH : Folder chứa file xuất ra. vd ./data   
* WORD_COUNT_PATH : Path đến file word_count.json
-use-preprocessor : Xét khi tạo bộ word_count có sử dụng preprocess hay không.

Để tạo bộ word_count, chứa số lượng từ xuất hiện. Chạy  
python main.py -create-word-count -csv-folder-path PATH_TO_CSV_FOLDER -vocabulary_size VOCABULARY_SIZE -save-path SAVE_PATH -use-preprocessor


Để tạo bộ word_mapper, chứa map N từ dùng làm từ điển cho traing. Chạy  
python main.py -create-mapper -csv-folder-path PATH_TO_CSV_FOLDER -vocabulary_size VOCABULARY_SIZE -save-path SAVE_PATH  

Khởi tạo word_mapper khi chưa có word_count thì word_count sẽ được tạo tự động và lưu lại.   
Trường hợp muốn tạo word_mapper từ file word_count có sẵn, chạy  
python main.py -create-mapper -csv-folder-path PATH_TO_CSV_FOLDER -vocabulary_size VOCABULARY_SIZE -word_count_path WORD_COUNT_PATH  


VD:  
python main.py -create-mapper -csv-folder-path ./data/*.csv -vocabulary_size 10000 -save-path ./data  
Output: ./data/word_mapper.json  + ./data/word_count.json

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
Output: Model chứa thông tin word2vec + word_embedding.vec chứa feature vector theo định dạng word2vec của google ( word 0.0 0.1 ... \n word 0.2)   

Để build bộ word_embedding chứa các feature vector từ model đã train sẵn, chạy:  
python main.py -create-embedding -save-path SAVE_PATH -mapper-path MAPPER_PATH   
VD:  
python main.py -create-embedding -save-path "./temp/shortdata" -mapper-path "./temp/word_mapper.json"  


!! Nếu quá trình train bị dừng lại giữa chừng, tiếp tục chạy lại trên save folder đó, chương trình sẽ tự động resume theo file progress.json  

