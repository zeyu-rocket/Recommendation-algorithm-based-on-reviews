# 基于社交评论分析的个性化推荐算法研究

## 摘要

### ABSTRACT

## 1. 绪论

本文主要使用Bert模型对文本进行情感分析，随后协同过滤，筛选出兴趣爱好相似的用户。首先对Goodreads网站的数据集进行数据清洗，筛选出用户id，书名，用户评论。随后根据这份数据里的用户评论，使用来自face hugging的Bert预训练模型，分析出其情感倾向评分。随后根据这份情感倾向进行邻近算法研究，生成推荐模型，根据用户id对其书籍做出推荐。

### 1.1 课题研究背景和意义

文本分析作为目前已应用推荐系统的辅助。文本信息挖掘较为关键，稀疏性问题较难解决。从评论分析用户与物品的关系会受到许多干扰项。

### 1.2 课题研究的热点和发展现状

自然语言处理；情感分析；DeepCoNN模型（清华）；协同过滤（较多）；NARRE（清华）；

### 1.3 文献综述

DeepCoNN;NARRE

## 2 模型假设与符号说明

### 2.1 模型假设

- 假设预训练模型能够分析评论文本情感。

### 2.2 符号说明

## 3 Bert模型应用



### 3.1 模型描述

### 3.2 数据描述与数据预处理

### 3.3 数学处理与建立过程

### 3.4 模型结果

## 4 模型建立（协同过滤）

### 4.1 模型描述

### 4.2 数据描述与数据预处理

### 4.3 数学处理与建立过程

### 4.4 模型结果

## 5 讨论

### 5.1 局限性与改进空间

### 5.2 敏感性分析

## 6 结论

### 6.1 总结与未来研究方向

## 8 参考文献

## 9 附录

为了符合这条规则，你应该确保你的 Markdown 文件在最后一行内容后正好以一个换行字符结束，没有尾随的空格或制表符。这通常可以通过配置文本编辑器或 IDE 在保存文件时自动插入文件末尾的换行符来实现。