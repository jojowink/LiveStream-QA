import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import spacy

nlp = spacy.load('zh_core_web_sm')  # NLP中文模型


# 转换弹幕txt文档为dataframe格式
def change_txt_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_time = None
    for line in lines:
        line = line.strip()
        if re.match(r'\d{2}:\d{2}', line):  # 匹配时间戳
            current_time = line
        elif "：" in line:  # 用户名 + 内容
            user, content = line.split("：", 1)
            data.append({
                "timestamp": current_time,
                "user": user,
                "content": content.strip()
            })

    return pd.DataFrame(data)


# NLP加规则法检测弹幕问题
def is_question(content):
    question_keywords = ["为什么", "能否", "如何", "多久", "是不是", "吗", "?", "会不会", "可以不", "咋", "几", "还是",
                         "多长", "的吧", "能不能", "哪", "怎么"]
    transition_words = ["但是", "不过", "结果", "最终"]
    negative_words = ["没", "不", "无法", "只为了", "仅仅"]
    if any(keyword in content for keyword in question_keywords):
        return True
    # 陈述性问题识别方法
    if any(word in content for word in negative_words) or any(word in content for word in transition_words):
        return True
    if re.search(r"[\?？]$", content):
        return True
    doc = nlp(content)
    if any(token.tag_ in ["WRB", "WP"] for token in doc):  # WRB, WP 为问句特征
        return True
    return False


# 通过用户名判断是否为回答
def is_answer(user, content):
    return user == "田博士带娃学习" and len(content) > 0


# 知识库预处理
def change_knowledgebase(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    knowledge_base = []
    question, answer = None, None

    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\..+", line):
            if question and answer:
                knowledge_base.append({"question": question, "answer": answer})
            question = line.split('.', 1)[1].strip()
            answer = None  # 重置答案
        elif line:  # 非空行且不是问题，视为答案
            answer = line if answer is None else f"{answer} {line}"
        else:
            continue
    if question and answer:
        knowledge_base.append({"question": question, "answer": answer})

    return pd.DataFrame(knowledge_base)


# 知识库模板向量化
def vectorize_template(template_df, model_name):
    model = SentenceTransformer(model_name)
    template_df['question_vector'] = list(
        model.encode(template_df['question'].tolist(), convert_to_tensor=True).cpu().numpy())
    template_df['answer_vector'] = list(
        model.encode(template_df['answer'].tolist(), convert_to_tensor=True).cpu().numpy())
    return template_df


def save_vectorize_template(template_df, output_path):
    template_df.to_pickle(output_path)


if __name__ == '__main__':
    file_path = './data/danmu/田博士 20241127 早场.txt'
    template_path = './data/Template/直播间常见问题.txt'
    vectorize_template_path = './out/vectorizedTemplate/直播间常见问题.pkl'
    model_name = './model/all-MiniLM-L6-v2/'
    df = change_txt_to_dataframe(file_path)
    df['is_question'] = df['content'].apply(is_question)
    df['is_answer'] = df.apply(lambda row: is_answer(row['user'], row['content']), axis=1)
    print(df)
    template_df = change_knowledgebase(template_path)
    print("Parsed Knowledge Base:")
    print(template_df)
    template_vectorized = vectorize_template(template_df, model_name)
    print("Vectorized Knowledge Base:")
    print(template_vectorized)
    save_vectorize_template(template_vectorized, vectorize_template_path)
    print(f"Vectorized knowledge base saved to {vectorize_template_path}.")
