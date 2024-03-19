import streamlit as st
import pandas as pd
import random


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
