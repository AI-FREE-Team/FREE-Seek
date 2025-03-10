# FREE-Seek


# Llama-3.2-3B-Traditional-Chinese-FREE-Seek <a href="https://colab.research.google.com/github/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/blob/master/Data_Deployment_colab.ipynb"><img src="https://img.shields.io/badge/%E5%AF%A6%E4%BD%9C-Colab-yellow.svg?style=popout-square" alt="範例 Colab"></a>
# Llama-3.2-3B Deepseek 蒸餾繁體中文模型：FREE-Seek

![人工智慧 - 自由團隊](https://raw.githubusercontent.com/chenkenanalytic/img/master/af/aifreeteam.png)

## Preface 前言

在 DeepSeek 掀起全球一陣對於推理模型的風潮後，各家學研機構、企業都紛紛投入研究，在台灣也有不少 LLM 領域的技術先進加入此戰局，但願意將其數據、模型進行開源的開發團隊卻少之少，因此在投入推理模型研究的路上， AI . FREE Team 自詡能拋磚引玉，提供台灣、繁體中文社群我們團隊初步的研究成果：FREE-Seek，即使模型仍有許多改善面向與空間，但希望能透過此開源專案，活絡台灣在開源 LLM 的技術研究。

<br>

## Training Data Details 訓練資料說明

FREE-Seek 模型使用 Llama-3.2-3B 進行兩階段訓練，包含：(1)預訓練(Pretraining) (2)蒸餾訓練(Distill-SFT)，

在(1)預訓練中，在繁體中文網站中進行各式新聞、各大網路社群的純文本爬蟲，訓練資料量約為 5GB 繁體中文資料集，

在(2)蒸餾訓練中，主要採行兩種資料進行訓練：1. 使用 gpt-4o 合成資料訓練； 2.  使用繁體化的 Deepseek 數據集。

<br>

## Updates 更新紀錄

 - 2025.03.10 增加 Repo 說明、Colab 運行程式碼
 - 2025.03.03 上傳 FREE-Seek V1 模型到 HuggingFace (<a href='https://huggingface.co/AI-FREE-Team/Llama-3.2-3B-Traditional-Chinese-FREE-Seek'>FREE-Seek</a>)
<br>

## Datasets 資料集

 - 預訓練資料集 (暫不提供)

 - 蒸餾資料集 (<a href='https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k'>開源簡中數據集</a>、繁中化數據集-整理中)


<br>

## Usage 使用方法

### 1. 完整資料集 - whole Dataset (13,065 characters)
``` bash
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 確保使用的是 FREE-Seek 的模型
MODEL_NAME = "AI-FREE-Team/Llama-3.2-3B-Traditional-Chinese-FREE-Seek"

# 設定裝置 (使用 CUDA 或 CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
```
 
## Issues 問題與發現

1. 現行模型僅能進行單輪問答，可能原因為： (1)訓練資料 token 數量分佈多在2k-4k，SFT階段max-token設定在4k (2)3B小模型參數量限制。
2. 現行模型未進行"護欄"訓練，模型可能產生不適當的回覆，請使用者切勿用於非法用途。

(若有其他發現，歡迎使用者隨時跟我們 Updates)


## Citing
```
@misc{AI.FREE2025,
  author = {Po-Chuan Chen},
  title = {Llama-3.2-3B-Traditional-Chinese-FREE-Seek},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/AI-FREE-Team/Llama-3.2-3B-Traditional-Chinese-FREE-Seek}},
}
```
 
<br> 

