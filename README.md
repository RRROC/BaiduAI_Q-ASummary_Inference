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



