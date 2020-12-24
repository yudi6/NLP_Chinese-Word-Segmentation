# Readme

## 代码环境

- Pytorch版本：1.7.0+cu110

- 运行环境：Windows 10

## 运行说明

### 1、训练

- 训练时运行`main.py`文件即可进行训练，在`main.py`文件中**TrainWord2Vec**函数用于选择训练用的词向量文件，函数参数`load_word_model`有三种输入'baidu'、'merge'、'Word2Vec'，对应使用的三种词向量。

  
- 若选择'Word2Vec'词向量，如下。

  ```python
  word_weight = TrainWord2Vec(all_sentence, INPUT_SIZE, word_to_ix, 'Word2Vec')
  ```

- 训练其他参数可修改，最终模型保存到model文件夹中

### 2、测试

- 测试时运行`test_model.py`文件即可进行测试，在`test_model.py`文件中使用以下语句进行模型的读取，可进行修改。如果读取的测试集文件是分好词的，则终端会输出F1得分。

  ```python
  model = torch.load('model/model_merge.pkl')
  ```

- 最终结果输出到`result`文件夹中，采用的是‘a’读写模式。

  ```python
          with open('result/model_merge_result.txt', 'a', encoding='utf-8') as fp:
              fp.write(" ".join(model_result) + '\n')
  ```
