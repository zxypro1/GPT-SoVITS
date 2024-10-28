# Download moda ASR related models
from modelscope import snapshot_download
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',revision="v2.0.4")

# Download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip unzip and rename to G2PWModel, and then place them in GPT_SoVITS/text.

import os
import requests
import zipfile
import shutil

# 定义下载链接和目标路径
url = 'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip'
download_path = 'G2PWModel_1.1.zip'
target_dir = '../GPT_SoVITS/text'

# 下载文件
response = requests.get(url)
with open(download_path, 'wb') as file:
    file.write(response.content)

# 解压文件
with zipfile.ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall('.')

# 重命名解压后的文件夹
os.rename('G2PWModel_1.1', 'G2PWModel')

# 移动文件夹到目标目录
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
shutil.move('G2PWModel', target_dir)

# 清理临时文件
os.remove(download_path)

print("下载、解压、重命名和移动操作完成。")