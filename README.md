# 网络舆情研究组技术练习
![](image.png)  
 ***仅用于组内学习、交流***  
 

## 练习任务（不定期增加）
### 机器学习任务
* 泰坦尼克号乘客生存预测  
*[参考链接：sklearn](https://scikit-learn.org/stable/#)*    
  * 数据预处理  
  * 使用SVM, LR, GBDT, DT, RF等机器算法完成预测  
  * 使用网格搜索法对相关参数调优
  * 绘制ROC曲线，输出AUC, P, R, F值  
  * 十折交叉验证  
### 词表示任务
* 词向量  
*参考链接：[word2vec](https://radimrehurek.com/gensim/)*  
  * 文本分词、去停用词  
  * 利用大规模领域文本训练skip-gram, CBOW词向量模型
  * 利用训练好的词向量模型表示文本，并生成pickle文件保存
* 上下文词向量  
*参考链接：[BERT-keras](https://github.com/CyberZHG/keras-bert)*
  * 利用BERT表示用于分类的文本  
* 参考文献  
  * [Mikolov et al. (2013), Distributed-representations-of-words-and-phrases-and-their-compositionality](/references/distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
  * [Devlin et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](/references/bert.pdf)

### 文本情感分析任务
* BiLSTM情感预测  
* CNN情感预测    
* BiLSTM-CNN情感预测  
* BiLSTM-Attention情感预测  
* BiLSTM-CRF情感预测
### 图片情感分析任务
* 图片情感识别  
### 多模态融合情感分析任务
* 融合文本和图片进行情感识别
