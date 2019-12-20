![](image.png)  
# 网络舆情研究组技术练习
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
### 文本表示任务
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
*参考链接:[keras](https://keras.io/),[pytorch](https://pytorch.org/docs/stable/index.html)*  
* BiLSTM情感预测  
  * 参数调优，包括：units，dropout，batch_size等  
  * 输出classification_report  
  * 训练过程中准确率和损失值的可视化图  
* CNN情感预测  
  * 参数调优
  * 输出classification_report
  * 训练过程中准确率和损失值的可视化图 
* BiLSTM-CNN情感预测  
  * 要求同上  
* BiLSTM-Attention情感预测  
  * 应用的attention机制包括self-attention，multi-head attention
  * *对应用的注意力机制的文本进行可视化（选做）*  
* BiLSTM-CRF情感预测
  * 要求同BiLSTM  
* 参考文献
  * [Young et al.(2018), Recent Trends in Deep Learning Based Natural Language Processing](https://arxiv.xilesou.top/pdf/1708.02709)  
### 图片情感分析任务
* 图片分类  
  * 利用预训练VGG16/19对花卉种类进行分类
  * 微调（fine-tune）网络
  * 获取中间层输出
  * 对不同的卷积层、池化层输出进行可视化
* 图片情感分析
* 参考文献
  * [Campos et al. (2017), From pixels to sentiment: Fine-tuning CNNs for visual sentiment prediction](references/from-pixel.pdf)
### 多模态融合情感分析任务
* 融合文本和图片进行情感识别
