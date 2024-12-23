import os
from datetime import datetime, timedelta

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
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
    negative_words = ["没", "无法", "只为了", "仅仅"]
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


# 使用语义匹配模型来匹配弹幕问题和答案
def match_qa(df, model_name, timestamp_col='timestamp', question_col='is_question', answer_col='is_answer',
             content_col='content', user_col='user',
             similarity_threshold=0.5, window_minutes=5):
    model = SentenceTransformer(model_name)
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%H:%M')
    questions_df = df[df[question_col]].copy()
    answers_df = df[df[answer_col]].copy()

    if questions_df.empty or answers_df.empty:
        print("No questions or answers found")
        return pd.DataFrame(
            columns=['Question', 'Answer', 'Similarity', 'Question_User', 'Answer_User', 'Question_Time',
                     'Answer_Time'])

    matched_pairs = []

    # 遍历每个问题
    for _, question_row in questions_df.iterrows():
        question_text = question_row[content_col]
        question_time = question_row[timestamp_col]
        question_user = question_row[user_col]

        # 筛选符合时间窗口的答案
        valid_answers_df = answers_df[
            (answers_df[timestamp_col] >= question_time) &
            (answers_df[timestamp_col] <= question_time + timedelta(minutes=window_minutes))
            ]

        if valid_answers_df.empty:
            # 没有匹配的回答，记录问题
            matched_pairs.append({
                "user": question_user,
                "Question": question_text,
                "Answer": "",
                "Similarity": None,
                "Question_Time": question_time,
                "Answer_Time": None
            })
            continue

        valid_answers_df = valid_answers_df.reset_index(drop=True)

        # 优先查找@的回答
        valid_answers_df[content_col] = valid_answers_df[content_col].str.replace(r'\s+', ' ', regex=True).str.strip()
        question_user = question_user.strip()
        mention_pattern = f"@{re.escape(question_user)}"
        mentioned_answers = valid_answers_df[
            valid_answers_df[content_col].str.contains(mention_pattern, na=False)
        ]
        if not mentioned_answers.empty:
            # 直接取第一个 @ 回答
            first_mentioned_answer = mentioned_answers.iloc[0]
            matched_pairs.append({
                "user": question_user,
                "Question": question_text,
                "Answer": first_mentioned_answer[content_col],
                "Similarity": 1.0,  # 高置信度
                "Question_Time": question_time,
                "Answer_Time": first_mentioned_answer[timestamp_col]
            })
            continue  # 已找到直接回答，跳过语义匹配

        # 计算问题与有效答案的语义相似度
        question_embedding = model.encode([question_text], convert_to_tensor=True)
        valid_answers_texts = valid_answers_df[content_col].tolist()
        answer_embeddings = model.encode(valid_answers_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_embedding, answer_embeddings)
        # 找到最相似的答案
        best_match_found = False
        for answer_idx, similarity_score in enumerate(similarities[0]):
            similarity_score = similarity_score.item()
            if "@" in valid_answers_texts[answer_idx] and not re.search(mention_pattern,
                                                                        valid_answers_texts[answer_idx]):
                continue
            if similarity_score > similarity_threshold:
                matched_pairs.append({
                    "user": question_user,
                    "Question": question_text,
                    "Answer": valid_answers_texts[answer_idx],
                    "Similarity": similarity_score,
                    "Question_Time": question_time,
                    "Answer_Time": valid_answers_df.iloc[answer_idx][timestamp_col]
                })
                best_match_found = True
                break

        # 如果没有找到合适的回答，则留空回答
        if not best_match_found:
            matched_pairs.append({
                "user": question_user,
                "Question": question_text,
                "Answer": "",  # 空回答
                "Similarity": None,  # 没有相似度
                "Question_Time": question_time,
                "Answer_Time": None  # 没有回答时间
            })

    # 返回匹配结果 DataFrame
    return pd.DataFrame(matched_pairs)


# 与知识库相匹配
def validate(matchqa_df, vectorized_template_path, model_name):
    # 加载向量化后的知识库模板
    knowledge_base = pd.DataFrame()
    for file in os.listdir(vectorized_template_path):
        if file.endswith('.pkl'):
            file_path = os.path.join(vectorized_template_path, file)
            temp_knowledge_base = pd.read_pickle(file_path)
            temp_knowledge_base['source_file'] = os.path.splitext(file)[0]  # 标记来源文件
            knowledge_base = pd.concat([knowledge_base, temp_knowledge_base], ignore_index=True)

    # 转换 question_vector 和 answer_vector 为 PyTorch 张量
    knowledge_base['question_vector'] = knowledge_base['question_vector'].apply(torch.tensor)
    knowledge_base['answer_vector'] = knowledge_base['answer_vector'].apply(torch.tensor)

    # 模型加载（用于生成未匹配到模板时的新向量）
    model = SentenceTransformer(model_name)

    # 初始化结果存储
    results = []

    # 遍历问答对 DataFrame
    for _, row in matchqa_df.iterrows():
        user = row['user']
        question = row['Question']
        answer = row['Answer']
        question_time = row['Question_Time']
        answer_time = row['Answer_Time']

        # 提取场控回答的向量
        question_vector = model.encode([question], convert_to_tensor=True)
        answer_vector = model.encode([answer], convert_to_tensor=True)

        # 计算与知识库问题的相似度
        kb_question_vectors = torch.stack(knowledge_base['question_vector'].tolist())
        question_similarities = util.pytorch_cos_sim(question_vector, kb_question_vectors)[0]

        # 找到最相似的模板问题
        max_question_idx = torch.argmax(question_similarities).item()
        max_question_similarity = question_similarities[max_question_idx]

        # 提取对应模板答案的向量
        kb_answer_vector = knowledge_base.iloc[max_question_idx]['answer_vector']
        answer_similarity = util.pytorch_cos_sim(answer_vector, kb_answer_vector).item()

        # 判断是否匹配成功
        is_correct = max_question_similarity >= 0.5 and answer_similarity >= 0.5

        # 获取模板内容
        matched_question = knowledge_base.iloc[max_question_idx]['question']
        matched_answer = knowledge_base.iloc[max_question_idx]['answer']

        # 计算回答延迟
        delay = None
        if pd.notnull(answer_time):
            delay = (pd.to_datetime(answer_time) - pd.to_datetime(question_time)).total_seconds()

        # 保存结果
        results.append({
            '质检表ID': len(results) + 1,
            '用户名': user,
            '片段内容': f"{question}\n{answer}",
            '对话原文': question,
            '知识库原文': f"{matched_question} -> {matched_answer}",
            '是否回答正确': "是" if is_correct else "否",
            '回答延迟': delay,
            '问题类别': None
        })

    # 转为 DataFrame 并保存
    return pd.DataFrame(results)


if __name__ == '__main__':
    file_path = input("请输入要处理的文件路径：").strip()
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
    else:
        model_name = '../model/all-MiniLM-L6-v2'
        vectorized_template_path = '../out/vectorizedTemplate'
        output_path = '../out/results'
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        print("正在处理弹幕数据...")
        df = change_txt_to_dataframe(file_path)
        df.to_csv(f'../out/dataframe/{base_name}/{base_name}_preprocessing.csv', index=False, encoding='utf-8')

        df['is_question'] = df['content'].apply(is_question)
        df['is_answer'] = df.apply(lambda row: is_answer(row['user'], row['content']), axis=1)
        df.to_csv(f'../out/dataframe/{base_name}/{base_name}_qaDeterment.csv', index=False, encoding='utf-8')

        print("正在匹配问答对...")
        matchqa_df = match_qa(df, model_name)
        matchqa_df.to_csv(f'./out/dataframe/{base_name}/{base_name}_matchqa.csv', index=False, encoding='utf-8')
        print("正在与知识库匹配，生成结果表格...")
        results_df = validate(matchqa_df, vectorized_template_path, model_name)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{base_name}_report.xlsx")
        results_df.to_excel(output_file, index=False)
        print(f"验证结果已保存到: {output_file}")
        print("处理完成，结果已保存到相应目录。")
        input("Press any key to continue...")
