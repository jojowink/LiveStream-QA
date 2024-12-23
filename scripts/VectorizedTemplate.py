import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer


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
    input_folder = '../data/Template/'  # 输入文件夹路径
    output_folder = '../out/vectorizedTemplate/'  # 输出文件夹路径
    model_name = '../model/all-MiniLM-L6-v2/'  # 模型路径

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有txt文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            txt_file_path = os.path.join(input_folder, file_name)
            pkl_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.pkl")

            # 检查是否已经向量化
            if os.path.exists(pkl_file_path):
                print(f"已向量化，跳过文件: {file_name}")
                continue

            # 处理未向量化的文件
            print(f"正在处理文件: {file_name}")
            template_df = change_knowledgebase(txt_file_path)
            print("知识库解析完成")
            template_vectorized = vectorize_template(template_df, model_name)
            print("知识库向量化完成")
            save_vectorize_template(template_vectorized, pkl_file_path)
            print(f"向量化结果已保存为: {pkl_file_path}")

    print("所有文件处理完成！")
