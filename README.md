# BaiduAI_Q-ASummary_Inference

## Summary

> 汽车大师问答摘要与推理。要求使用汽车大师提供的11万条 技师与用户的多轮对话与诊断建议报告 数据建立模型，模型需基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，综合考验模型的归纳总结与推断能力。
>
> 汽车大师是一款通过在线咨询问答为车主解决用车问题的APP，致力于做车主身边靠谱的用车顾问，车主用语音、文字或图片发布汽车问题，系统为其匹配专业技师提供及时有效的咨询服务。由于平台用户基数众多，问题重合度较高，大部分问题在平台上曾得到过解答。重复回答和持续时间长的多轮问询不仅会花去汽修技师大量时间，也使用户获取解决方案的时间变长，对双方来说都存在资源浪费的情况。
>
> 为了节省更多人工时间，提高用户获取回答和解决方案的效率，汽车大师希望通过利用机器学习对平台积累的大量历史问答数据进行模型训练，基于汽车大师平台提供的历史多轮问答文本，输出完整的建议报告和回答，让用户在线通过人工智能语义识别即时获得全套解决方案

项目具体介绍 --> [百度AI-Studio-常规赛：问答摘要与推理](https://aistudio.baidu.com/aistudio/competition/detail/3)



## Week-1

This program aims to train a AI to generate Q & A summary and Inference according to the article.

Currently the first step, that is, building vocabulary table has been done.

Based on the train data and test data, all words are given a index by using Jieba and Pandas

Please run the program from Main_Entrance.py and the vocabulary table called "vocab.txt" will be generated in the output folder.



## Week-2

Create a file called Vocab_build.py for generating word embedding by using Gensim.

After that, by using the vocabulary created last week, a vocabulary metric has been created.

Besides, I also tried fastText to train word vector, and compared with word2vec model.



## Week-3

This week, the baseline of the whole system has been finished. Basically it adopts Seq2Seq model to analyze and train data.

It contains encoder layer, decoder layer and attention layer. The size of vocabulary is limited to 30000 in order to boost training speed.  



## Week-4

The testing part has been added. Now it can restore the checkpoint and generate the summary finally. The baseline has been completed.

Now it is using greedy_search in the testing part. I will add the beam search in the future (in process).

Due to the capacity of local GPU, the vocabulary size is limited to 2000 in training section, which causes the model effect is not quite good.

I will try to run it in cloud and make it better.



## Week-5

The structure of the program has been improved. Now the program added PGN model, and it is separated from Seq2Seq model. You can use the parameters called "model" and "mode" to control which model you want to use (Seq2Seq or PGN) and testing or training.

Basically, this update mainly focuses on the PGN. It also contains Beam Search and Coverage to solve the problems of OOV words and repeating words.

Due to the limitation of my PC's capacity, I restricted the vocab_size to 2000. It may get better performance if the full vocabulary can be used.

