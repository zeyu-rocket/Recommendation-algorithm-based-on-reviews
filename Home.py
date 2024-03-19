import streamlit as st
import joblib
import pandas as pd
import random


# 加载矩阵
# 创建csr矩阵做协同过滤
import pandas as pd
from scipy.sparse import csr_matrix

# 替换为你的CSV文件路径
file_path = "pivot_use.csv"

# 使用pandas读取CSV文件
df = pd.read_csv(file_path)

# 将用户ID和书名映射为整数索引
df["user_id_idx"] = df["user_id"].astype("category").cat.codes
df["book_title_idx"] = df["title_without_series"].astype("category").cat.codes

# 确保所有索引都是非负的
assert df["user_id_idx"].min() >= 0, "user_id_idx contains negative values"
assert df["book_title_idx"].min() >= 0, "book_title_idx contains negative values"

# 创建CSR矩阵
ratings_csr = csr_matrix(
    (df["bert_rating"], (df["user_id_idx"], df["book_title_idx"])),
    shape=(df["user_id_idx"].max() + 1, df["book_title_idx"].max() + 1),
)


# 加载推荐函数
def recommend_for_user(user_id_idx, ratings_csr, model_knn, n_recommendations=10):
    # 查询这个用户的k个最近邻居
    distances, indices = model_knn.kneighbors(
        ratings_csr[user_id_idx], n_neighbors=n_recommendations + 1
    )

    # 返回最近邻居的索引和距离
    # 注意我们跳过第一个最近邻（即用户自己）
    return indices[0][1:], distances[0][1:]


# 加载模型和映射
model_knn = joblib.load("model_knn.pkl")
user_id_to_idx = joblib.load("user_id_to_idx.pkl")
book_idx_to_title = joblib.load("book_idx_to_title.pkl")

# Streamlit界面代码
st.title("书籍推荐系统")

# 假设ratings_csr也需要被加载或在这里以某种方式被构建

user_id_input = st.text_input("请输入您的用户ID:", "")

if user_id_input:
    user_id_idx = user_id_to_idx.get(user_id_input, None)
    if user_id_idx is not None:
        indices, distances = recommend_for_user(user_id_idx, ratings_csr, model_knn)
        recommended_books = [
            book_idx_to_title.get(idx, "Unknown Book") for idx in indices
        ]
        st.write("为您推荐的书籍:")
        for book in recommended_books:
            st.write(book)
    else:
        st.write("未找到用户ID，请确保输入正确。")


# 加载CSV文件
@st.cache_data  # 使用experimental_memo来优化性能，避免每次互动都重新加载文件
def load_data(filepath):
    return pd.read_csv(filepath)


# 随机选取user_id和title_without_series
def get_random_user_info(df):
    random_row = random.randint(0, len(df) - 1)
    user_id = df.iloc[random_row]["user_id"]
    title_without_series = df.iloc[random_row]["title_without_series"]
    return user_id, title_without_series


# 主函数
def main():
    # 设置页面标题
    st.title("随机显示用户ID和书籍标题")

    # 加载数据
    filepath = (
        "/Users/zeyu/Documents/GitHub/sentiment/pivot_use.csv"  # 替换成你的CSV文件路径
    )
    df = load_data(filepath)

    # 如果按钮被点击，则显示一个随机的user_id和title_without_series
    if st.button("显示随机用户信息"):
        user_id, title_without_series = get_random_user_info(df)
        st.write(f"随机选择的用户ID是: {user_id}")
        st.write(f"对应的书籍标题是: {title_without_series}")


if __name__ == "__main__":
    main()
