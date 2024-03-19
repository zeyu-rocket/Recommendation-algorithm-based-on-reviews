# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import requests
# from bs4 import BeautifulSoup
# import re

# # from transformers import AutoConfig

# # config = AutoConfig.from_pretrained(
# #     "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/nlptown:bert-base-multilingual-uncased-sentiment/config.json"
# # )

# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# # 指定模型和tokenizer的路径
# model_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/nlptown:bert-base-multilingual-uncased-sentiment"  # 替换为你的模型目录路径

# # 加载tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_path)

# # 加载模型
# model = BertForSequenceClassification.from_pretrained(model_path)


# tokens = tokenizer.encode("this is ok", return_tensors="pt")
# print(tokens)
# print(tokenizer.decode(tokens[0]))
# result = model(tokens)
# print(result)
# print(result.logits)
# score = int(torch.argmax(result.logits)) + 1
# print(score)


# # tokenizer = AutoTokenizer.from_pretrained(
# #     "nlptown/bert-base-multilingual-uncased-sentiment"
# # )

# # model = AutoModelForSequenceClassification.from_pretrained(
# #     "nlptown/bert-base-multilingual-uncased-sentiment"
# # )


import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm  # 导入tqdm

# 指定模型和tokenizer的路径
model_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/nlptown:bert-base-multilingual-uncased-sentiment"  # 修改为您的模型目录路径

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


# def score_review(review_text):
#     tokens = tokenizer.encode(review_text, return_tensors="pt")
#     result = model(tokens)
#     score = int(torch.argmax(result.logits)) + 1
#     return score


def score_review(review_text):
    # 限制Token序列的最大长度为512，并启用自动截断
    tokens = tokenizer.encode(
        review_text, return_tensors="pt", max_length=512, truncation=True
    )
    result = model(tokens)
    score = int(torch.argmax(result.logits)) + 1
    return score


def process_json_file(file_path):
    # 读取JSON文件
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    # 使用tqdm显示进度条
    for item in tqdm(data, desc="Processing Reviews"):
        review_text = item["review"]
        bert_rating = score_review(review_text)
        item["bert_rating"] = bert_rating

    # 将结果写入新的JSON文件
    output_file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/bert_reviews.json"  # 修改为您的输出文件路径
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write("\n")


# 调用函数，处理您的JSON文件
file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/reviews.json"  # 修改为您的输入文件路径
process_json_file(file_path)
