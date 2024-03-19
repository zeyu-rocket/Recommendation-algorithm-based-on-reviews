# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# from tqdm import tqdm

# # Specify the model and tokenizer path
# model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

# # Load the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)


# def score_review(review_text):
#     # Limit the token sequence length to 512 and enable automatic truncation
#     tokens = tokenizer.encode(
#         review_text, return_tensors="pt", max_length=512, truncation=True
#     )
#     result = model(tokens)
#     score = int(torch.argmax(result.logits)) + 1
#     return score


# def process_csv_file(file_path, output_file_path):
#     # Read the CSV file
#     df = pd.read_csv(file_path)

#     # Apply the score_review function to each review in the DataFrame
#     df["bert_rating"] = df["review_text"].apply(lambda x: score_review(x))

#     # Write the DataFrame with the new column to a new CSV file
#     df.to_csv(output_file_path, index=False)


# # Specify your input and output file paths
# input_file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/merged_file.csv"  # Change this to your input file path
# output_file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/merged_bertrating_file.csv"  # Change this to your desired output file path

# # Call the function to process your CSV file
# process_csv_file(input_file_path, output_file_path)


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

# 指定模型和tokenizer的路径
model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


def score_review(review_text):
    # 检查评论文本是否为空或NaN
    if pd.isna(review_text):
        # 如果是空或NaN，可以选择返回一个默认的评分，例如3，或者其他适当的处理方式
        return 3
    tokens = tokenizer.encode(
        review_text, return_tensors="pt", max_length=512, truncation=True
    )
    result = model(tokens)
    score = int(torch.argmax(result.logits)) + 1
    return score


def process_csv_file(file_path, output_file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 使用tqdm显示进度条，并对每条评论应用score_review函数
    tqdm.pandas(desc="Processing Reviews")
    df["bert_rating"] = df["review_text"].progress_apply(lambda x: score_review(x))

    # 将DataFrame写回到新的CSV文件，包含新增的bert_rating列
    df.to_csv(output_file_path, index=False)


# 指定你的输入和输出文件路径
input_file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/merged_file.csv"  # Change this to your input file path
output_file_path = "/Users/zeyu/Documents/Learning/若楠的毕设/sentiment/merged_bertrating_file.csv"  # Change this to your desired output file path

# 调用函数处理CSV文件
process_csv_file(input_file_path, output_file_path)
