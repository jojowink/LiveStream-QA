import pandas as pd
import re


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
                "content": content.strip(),
                "is_question": bool(re.search(r'[？?]', content)),  # 简单判断是否为问题
                "is_answer": "@" in content
            })

    return pd.DataFrame(data)


if __name__ == '__main__':
    file_path = './danmu/田博士 20241127 早场.txt'
    df = change_txt_to_dataframe(file_path)
    print(df)
