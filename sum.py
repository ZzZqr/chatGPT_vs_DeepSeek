import pandas as pd

data_all = pd.read_excel("./data.xlsx", sheet_name=None)
hard_to_heal = []
amputation = []
data_case = data_all["200description"]

import os

# 替换为你的文件路径
folder = './output/'
output_folder = './output'

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

dsr1 = "Answer of DeepSeek R1:"
dsv3 = "Answer of DeepSeek V3:"
gpt4o = "Answer of ChatGPT-4o:"
o3 = "Answer of o3:"

# 遍历DataFrame的每一行
for idx, row in data_case.iterrows():
    # 获取描述
    description = row['English Case Description']

    # 构建读取文件路径 ds_r1_case{idx}.txt
    ds_r1_filename = f'ds_r1_case{idx+1}.txt'
    ds_r1_path = os.path.join(folder, ds_r1_filename)
    # 检查原文件是否存在
    if not os.path.exists(ds_r1_path):
        print(f'Warning: {ds_r1_path} not found, skipping...')
        continue
    # 读取ds_r1_case文件内容
    with open(ds_r1_path, 'r', encoding='utf-8') as f:
        ds_r1_content = f.read()


    # 构建读取文件路径 ds_r1_case{idx}.txt
    ds_v3_filename = f'ds_v3_case{idx + 1}.txt'
    ds_v3_path = os.path.join(folder, ds_v3_filename)
    # 检查原文件是否存在
    if not os.path.exists(ds_v3_path):
        print(f'Warning: {ds_v3_path} not found, skipping...')
        continue
    # 读取ds_r1_case文件内容
    with open(ds_v3_path, 'r', encoding='utf-8') as f:
        ds_v3_content = f.read()

    amputation

    gpt_filename = f'gpt_promptI_case{idx + 1}.txt'
    gpt_path = os.path.join(folder, gpt_filename)
    # 检查原文件是否存在
    if not os.path.exists(gpt_path):
        print(f'Warning: {gpt_path} not found, skipping...')
        continue
    # 读取ds_r1_case文件内容
    with open(gpt_path, 'r', encoding='utf-8') as f:
        gpt4o_content = f.read()

    o3_filename = f'o3_case{idx + 1}.txt'
    o3_path = os.path.join(folder, o3_filename)
    # 检查原文件是否存在
    if not os.path.exists(o3_path):
        print(f'Warning: {o3_path} not found, skipping...')
        continue
    # 读取ds_r1_case文件内容
    with open(o3_path, 'r', encoding='utf-8') as f:
        o3_content = f.read()

    # 构建新文件内容
    combined_content = f"{description}\n\n\n{dsr1}\n\n{ds_r1_content}\n\n\n{dsv3}\n\n{ds_v3_content}\n\n\n{gpt4o}\n\n{gpt4o_content}\n\n\n{o3}\n\n{o3_content}"

    # 写入新的文件 case_{idx}.txt
    output_path = os.path.join(output_folder, f'case_{idx+1}.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)

    print(f'Saved: {output_path}')