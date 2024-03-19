import json

# JSON文件路径
file_path = "bert_reviews.json"

# 初始化一个列表来存储解析后的JSON对象
reviews = []

# 尝试逐行打开并解析JSON文件
with open(file_path, "r") as file:
    for line in file:
        try:
            # 尝试解析当前行为JSON
            review = json.loads(line)
            # 添加到列表中
            reviews.append(review)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from line: {e}")

# 初始化一个列表来存储每条评论的差值
rating_differences = []

# 遍历每条评论，计算rating和bert_rating之间的差值
for review in reviews:
    difference = abs(review["rating"] - review["bert_rating"])
    rating_differences.append(difference)

# 输出差值列表和平均差值
print("Rating differences:", rating_differences)
average_difference = sum(rating_differences) / len(rating_differences)
print("Average difference:", average_difference)
