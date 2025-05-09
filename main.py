import csv
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from openai import OpenAI
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score


def ds_query(case):
    client = OpenAI(api_key="your deepseek key", base_url="https://api.deepseek.com")

    response1 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": case},
        ],
        stream=False
    )

    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt2},
            {"role": "user", "content": case},
        ],
        stream=False
    )

    response3 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt3},
            {"role": "user", "content": case},
        ],
        stream=False
    )
    answer_segmented = [response1.choices[0].message.content, response2.choices[0].message.content, response3.choices[0].message.content]

    return answer_segmented


def gpt_query(case):
    client = OpenAI(api_key="your gpt key")

    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": case},
        ],
        stream=False
    )

    response2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt2},
            {"role": "user", "content": case},
        ],
        stream=False
    )

    response3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt3},
            {"role": "user", "content": case},
        ],
        stream=False
    )
    answer_segmented = [response1.choices[0].message.content, response2.choices[0].message.content, response3.choices[0].message.content]

    return answer_segmented


prompt1 = ("Now you are a burns and plastics surgeon and here are the patients you see. Base on the situation of a patient in the case, you need to answer one question: Would this patient's diabetic foot wounds have healed more than 50% within 4 weeks with routine debridement and dressing changes? Please select one of the following options: Yes, No. "
           "Here is a case of 'Yes': A 26-year-old male with a 5 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the foot, measuring approximately 1*1 cm², classified as grade 2B according to the UTC system. The ulcer had been present for 24 weeks prior to clinic visit. The patient had a history of no notable complications. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 35.1 g/L, creatinine at 98.0 μmol/L, and C-reactive protein (CRP) at 1.2 mg/L. History of foot ulcer: none. "
           "Here is another case of 'Yes': A 59-year-old female with a 17 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the right heel, measuring approximately 8*6 cm², classified as grade 3c according to the UTC system. The ulcer had been present for 8 weeks prior to clinic visit. The patient had a history of hyperlipidemia; cardiovascular disease; neurological disease; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 36.1 g/L, creatinine at 32.0 μmol/L, and C-reactive protein (CRP) at 1.0 mg/L. History of foot ulcer: none."
           "There is a case of 'No': A 77-year-old male with a 1 months years history of type 2 diabetes mellitus, has not received insulin therapy. The patient presented with a foot ulcer located at the foot, measuring approximately 7*5 cm², classified as grade 3B according to the UTC system. The ulcer had been present for 4 weeks prior to clinic visit. The patient had a history of hypertension; hyperlipidemia; cardiovascular disease; neurological disease; peripheral neuropathy; peripheral neuropathy. Lifestyle history revealed alcohol consumption and smoking. Laboratory tests showed serum albumin at 20.2 g/L, creatinine at 67.1 μmol/L, and C-reactive protein (CRP) at 108.6 mg/L. History of foot ulcer: none. You only need to answer Yes or No without punctuation. "
           "Here is another case of 'No':  A 42-year-old male with a 16 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the toes, measuring approximately 2*6 cm², classified as grade 3B according to the UTC system. The ulcer had been present for 2 weeks prior to clinic visit. The patient had a history of diabetic nephropathy. Lifestyle history revealed smoking. Laboratory tests showed serum albumin at 33.9 g/L, creatinine at 133.0 μmol/L, and C-reactive protein (CRP) at 6.2 mg/L. History of foot ulcer: none. "
           "You only need to answer this question with 'Yes' or 'No' without punctuation or reason. ")

# prompt2 = "Now you are a burns and plastics surgeon and here are the patients you see. Base on the situation of a patient in the case, you need to answer one question: Can this patient avoid amputation under the multidisciplinary collaborative treatment at the diabetic foot center? Please select one of the following options: Yes, No. Here is a case of 'Yes':  A 70-year-old male with a 9 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the toes, measuring approximately 2*2 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 1year prior to clinic visit. The patient had a history of no notable complications. Here is a case of 'No': A 60-year-old male with a 10 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the toes and forefoot, measuring approximately 3*4 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 1 weeks prior to clinic visit. The patient had a history of hypertension; cardiovascular disease; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 23.5 g/L, creatinine at 111.0 μmol/L, and C-reactive protein (CRP) at 109.9 mg/L. History of foot ulcer: none. You only need to answer Yes or No without punctuation."
prompt2 = ("Now you are a burns and plastics surgeon and I am the patient with diabetes foot you are going to see. Base on my situation, you need to answer one question: Can I avoid amputation under the multidisciplinary collaborative treatment at the diabetic foot center? Please select one of the following options: Yes, No. "
           "Here is a case of 'Yes':  A 80-year-old female with a 20 years history of type 2 diabetes mellitus, has not received insulin therapy. The patient presented with a foot ulcer located at the toe to sole, measuring approximately 2*5 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 3 weeks prior to clinic visit. The patient had a history of cardiovascular disease; neurological disease; peripheral neuropathy; peripheral neuropathy; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 25.8 g/L, creatinine at 46.0 μmol/L, and C-reactive protein (CRP) at 91.9 mg/L. History of foot ulcer: none. "
           "Here is another case of 'Yes': A 65-year-old male with a 18 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the dorsum of foot, measuring approximately 3*4 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 1 weeks prior to clinic visit. The patient had a history of hypertension; cardiovascular disease; neurological disease; peripheral neuropathy; peripheral neuropathy; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 33.1 g/L, creatinine at 1052.8 μmol/L, and C-reactive protein (CRP) at 108.1 mg/L. History of foot ulcer: none. "
           "Here is a case of 'No': A 80-year-old female with a 20 years history of type 2 diabetes mellitus, has not received insulin therapy. The patient presented with a foot ulcer located at the toe to sole, measuring approximately 2*5 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 3 weeks prior to clinic visit. The patient had a history of cardiovascular disease; neurological disease; peripheral neuropathy; peripheral neuropathy; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 25.8 g/L, creatinine at 46.0 μmol/L, and C-reactive protein (CRP) at 91.9 mg/L. History of foot ulcer: none. "
           "Here is another case of 'No': A 65-year-old male with a 18 years history of type 2 diabetes mellitus, has been on insulin therapy. The patient presented with a foot ulcer located at the dorsum of foot, measuring approximately 3*4 cm², classified as grade 3D according to the UTC system. The ulcer had been present for 1 weeks prior to clinic visit. The patient had a history of hypertension; cardiovascular disease; neurological disease; peripheral neuropathy; peripheral neuropathy; peripheral neuropathy. Lifestyle history revealed no tobacco or alcohol use. Laboratory tests showed serum albumin at 33.1 g/L, creatinine at 1052.8 μmol/L, and C-reactive protein (CRP) at 108.1 mg/L. History of foot ulcer: none. "
           "You only need to answer this question with 'Yes' or 'No' without punctuation or reason. ")

prompt3 = "Now you are a burns and plastics surgeon and here are the patients you see. Base on the situation of a patient in the case, you need to answer one question: What is the treatment strategy for this patient? "

data_all = pd.read_excel("./data.xlsx", sheet_name=None)
hard_to_heal = []
amputation = []
data_case = data_all["200description"]
hard_to_heal_true = data_case['hard-to-heal'].tolist()
amputation_true = data_case['Amputation'].tolist()
words = []

for i in range(0, 192):
    case = data_case.loc[i, 'English Case Description']
    ans = ds_query(case)
    if ans[0] == "Yes":
        hard_to_heal.append(0)
    elif ans[0] == "No":
        hard_to_heal.append(1)
    else:
        print("hth")
        print(ans)

    if ans[1] == "Yes" or ans[1] == "Yes.":
        amputation.append(0)
    elif ans[1] == "No" or ans[1] == "No.":
        amputation.append(1)
    else:
        print("amp")
        print(ans)

    with open("output/ds_v3_case"+str(i+1)+".txt", "w", encoding="utf-8") as f:
        f.write(ans[2])


# conf_matrix = confusion_matrix(hard_to_heal_true, hard_to_heal)
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Hard-to-heal', 'Hard-to-heal'],
#                 yticklabels=['Not Hard-to-heal', 'Hard-to-heal'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title(f'Confusion Matrix of Hard-to-heal Prediction by DeepSeek-v3')
# plt.tight_layout()
# plt.savefig("confusion_matrix_hth_deepseekv3.png", dpi=300)
# plt.savefig("confusion_matrix_hth_deepseekv3.pdf")
# plt.show()
#
# conf_matrix = confusion_matrix(amputation_true, amputation)
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Amputation', 'Amputation'],
#                 yticklabels=['Not Amputation', 'Amputation'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title(f'Confusion Matrix of Amputation Prediction by DeepSeek-V3')
# plt.tight_layout()
# plt.savefig("confusion_matrix_amputation_deepseekv3.png", dpi=300)
# plt.savefig("confusion_matrix_amputation_deepseekv3.pdf")
# plt.show()
#
# accuracy1 = accuracy_score(hard_to_heal_true, hard_to_heal)
# precision1 = precision_score(hard_to_heal_true, hard_to_heal)
# recall1 = recall_score(hard_to_heal_true, hard_to_heal)
# f11 = f1_score(hard_to_heal_true, hard_to_heal)
# aucc1 = roc_auc_score(hard_to_heal_true, hard_to_heal)
#
# accuracy2 = accuracy_score(amputation_true, amputation)
# precision2 = precision_score(amputation_true, amputation)
# recall2 = recall_score(amputation_true, amputation)
# f12 = f1_score(amputation_true, amputation)
# aucc2 = roc_auc_score(amputation_true, amputation)
#
# with open(r'./result.csv', mode='a', newline='', encoding='utf8') as cfa:
#     wf = csv.writer(cfa)
#     data1 = ['deepseek-chat', 'hard-to-heal', accuracy1, precision1, recall1, f11, aucc1]
#     wf.writerow(data1)
#     data2 = ['deepseek-chat', 'amputation', accuracy2, precision2, recall2, f12, aucc2]
#     wf.writerow(data2)