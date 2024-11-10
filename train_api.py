import os
import sys
import traceback
from typing import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from tools.my_utils import load_audio, check_for_existance, check_details, clean_path
from config import python_exec, exp_root, is_half
import tempfile
import shutil
from subprocess import Popen
from tools import my_utils
import json
import yaml
from tools.asr.config import asr_dict
import torch
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader
import ffmpeg

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv
# determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

version="v2"
weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))
pretrained_sovits_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

pretrained_model_list = (pretrained_sovits_name[-int(version[-1])+2],pretrained_sovits_name[-int(version[-1])+2].replace("s2G","s2D"),pretrained_gpt_name[-int(version[-1])+2],"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large","GPT_SoVITS/pretrained_models/chinese-hubert-base")

_=''
for i in pretrained_model_list:
    if os.path.exists(i):...
    else:_+=f'\n    {i}'
if _:
    print("warning:",i18n('以下模型不存在:')+_)

_ =[[],[]]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):_[0].append(pretrained_gpt_name[i])
    else:_[0].append("")##没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):_[-1].append(pretrained_sovits_name[i])
    else:_[-1].append("")
pretrained_gpt_name,pretrained_sovits_name = _

SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
for root in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(root,exist_ok=True)
def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, device, is_half):
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
            func = BsRoformer_Loader
            pre_fun = func(
                model_path = os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                device = device,
                is_half=is_half
            )
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if(os.path.isfile(inp_path)==False):continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)

class UVRRequest(BaseModel):
    model_name: str  # UVR模型名称
    inp_dir: str  # 输入文件夹路径
    opt_dir_vocal: str  # 保存人声文件夹路径
    opt_dir_ins: str  # 保存伴奏文件夹路径
    agg: int = 0  # 人声提取激进程度，0-20的整数
    format0: str  # 文件格式。可选："wav", "flac", "mp3", "m4a"

@APP.post("/uvr")
async def uvr_convert(request: UVRRequest):
    request = request.dict()
    try:
        model_name = request.model_name
        inp_root = request.inp_dir
        save_root_vocal = request.opt_dir_vocal
        paths = ""
        save_root_ins = request.opt_dir_ins
        agg = request.agg
        format0 = request.format0
        device = device
        is_half = is_half

        if model_name not in uvr5_names:
            return JSONResponse(status_code=400, content={"error": f"model_name: {model_name} is not supported"})

        response = []
        async for info in uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, device, is_half):
            response.append(info)
        
        return JSONResponse(status_code=200, content={"opt_dir_ins": save_root_ins, "opt_dir_vocal": save_root_vocal})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"uvr_convert failed \n Exception: {str(e)}"})

class SliceRequest(BaseModel):
    inp_dir: str # 输入文件夹路径
    opt_dir: str # 输出文件夹路径
    threshold: int  # 阈值（db），音量小于这个值视作静音的备选切割点
    min_length: int  # 最小长度（ms），每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
    min_interval: int  # 最小间隔（ms），最短切割间隔
    hop_size: int  # 跳跃大小，决定怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
    max_sil_kept: int  # 最大保留静音（ms），切完后静音最多留多长
    _max: float  # 最大值（0.0-1.0），归一化后最大值多少
    alpha: float  # 阿尔法值（0.0-1.0），混多少比例归一化后音频进来
    n_process: int  # 使用进程数（1-12）

@APP.post("/slice_audio")
async def slice_audio(request: SliceRequest):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            inp = my_utils.clean_path(request.inp_dir)
            opt_root = my_utils.clean_path(request.opt_dir)
            # 创建输出文件夹
            os.makedirs(opt_root, exist_ok=True)
            
            ps_slice = []
            for i_part in range(request.n_process):
                cmd = f'"{python_exec}" tools/slice_audio.py "{inp}" "{opt_root}" {request.threshold} {request.min_length} {request.min_interval} {request.hop_size} {request.max_sil_kept} {request._max} {request.alpha} {i_part} {request.n_parts}'
                p = Popen(cmd, shell=True)
                ps_slice.append(p)
            
            for p in ps_slice:
                p.wait()
            
            # 假设处理后的文件名为 output.wav
            output_file_path = os.path.join(opt_root, "output.wav")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"error": "Output file does not exist"})
            
            return JSONResponse(status_code=200, content={"opt_dir": f"{opt_root}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Slicing failed. \n Exception: {str(e)}"})

class DenoiseRequest(BaseModel):
    inp_dir: str  # 输入文件夹路径
    opt_dir: str  # 输出文件夹路径

@APP.post("/denoise_audio")
async def denoise_audio(request: DenoiseRequest):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            inp = my_utils.clean_path(request.inp_dir)
            opt_root = my_utils.clean_path(request.opt_dir)
            # 创建输出文件夹
            os.makedirs(opt_root, exist_ok=True)
            
            # 构建命令
            cmd = f'"{python_exec}" tools/cmd-denoise.py -i "{inp}" -o "{opt_root}" -p {"float16" if is_half else "float32"}'
            p = Popen(cmd, shell=True)
            p.wait()
            
            # 假设处理后的文件名为 output.wav
            output_file_path = os.path.join(opt_root, "output.wav")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"error": "Output file does not exist"})
            
            return JSONResponse(status_code=200, content={"opt_dir": f"{opt_root}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Denoising failed \n Exception: {str(e)}"})

class ASRRequest(BaseModel):
    inp_dir: str  # 输入文件夹路径
    opt_dir: str  # 输出文件夹路径
    model: str  # 模型名称
    model_size: str  # 模型大小
    lang: str  # 语言
    precision: str  # 精度

@APP.post("/asr")
async def asr(request: ASRRequest):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            inp = my_utils.clean_path(request.inp_dir)
            opt_root = my_utils.clean_path(request.opt_dir)
            # 创建输出文件夹
            os.makedirs(opt_root, exist_ok=True)
            
            # 构建命令
            cmd = f'"{python_exec}" tools/asr/{asr_dict[request.model]["path"]} -i "{inp}" -o "{opt_root}" -s {request.model_size} -l {request.lang} -p {request.precision}'
            p = Popen(cmd, shell=True)
            p.wait()
            
            # 假设处理后的文件名为 output.txt
            output_file_path = os.path.join(opt_root, "output.txt")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"message": "Output file does not exist"})
            
            return JSONResponse(status_code=200, content={"opt_text_dir": f"{opt_root}", "opt_dir": f"{inp}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"ASR task failed \n Exception: {str(e)}"})

class OneClickRequest(BaseModel):
    inp_text: str  # 文本标注的路径，参考asr任务的输出
    inp_dir: str  # 输入音频文件夹路径，参考asr任务的输出
    opt_dir: str  # 输出结果文件夹
    gpu_numbers1a: str  # 文本获取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
    gpu_numbers1Ba: str  # SSL自监督特征提取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
    gpu_numbers1c: str  # 语义token提取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
    bert_pretrained_dir: str  # 预训练的中文BERT模型路径
    ssl_pretrained_dir: str  # 预训练的SSL模型路径
    pretrained_s2G_path: str  # 预训练的SoVITS-G模型路径

@APP.post("/one_click")
async def one_click(request: OneClickRequest):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            inp = my_utils.clean_path(request.inp_dir)
            opt_root = my_utils.clean_path(request.opt_dir)
            # 创建输出文件夹
            os.makedirs(opt_root, exist_ok=True)
            
            # 处理逻辑
            inp_text = my_utils.clean_path(request.inp_text)
            if check_for_existance([inp_text, inp], is_dataset_processing=True):
                check_details([inp_text, inp], is_dataset_processing=True)
            opt_dir = opt_root
            os.makedirs(opt_dir, exist_ok=True)
            
            ps1abc = []
            # 1a
            path_text = f"{opt_dir}/2-name2text.txt"
            if not os.path.exists(path_text) or (os.path.exists(path_text) and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp,
                    "exp_name": opt_dir,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": request.bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names = request.gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update({
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    })
                    os.environ.update(config)
                    cmd = f'"{python_exec}" GPT_SoVITS/prepare_datasets/1-get-text.py'
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                opt = []
                for i_part in range(all_parts):
                    txt_path = f"{opt_dir}/2-name2text-{i_part}.txt"
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
            
            ps1abc = []
            # 1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp,
                "exp_name": opt_dir,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": request.ssl_pretrained_dir,
            }
            gpu_names = request.gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update({
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                })
                os.environ.update(config)
                cmd = f'"{python_exec}" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            for p in ps1abc:
                p.wait()
            
            ps1abc = []
            # 1c
            path_semantic = f"{opt_dir}/6-name2semantic.tsv"
            if not os.path.exists(path_semantic) or (os.path.exists(path_semantic) and os.path.getsize(path_semantic) < 31):
                config = {
                    "inp_text": inp_text,
                    "exp_name": opt_dir,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": request.pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = request.gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update({
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    })
                    os.environ.update(config)
                    cmd = f'"{python_exec}" GPT_SoVITS/prepare_datasets/3-get-semantic.py'
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = f"{opt_dir}/6-name2semantic-{i_part}.tsv"
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
            
            ps1abc = []
            return JSONResponse(status_code=200, content={"opt_dir": f"{opt_root}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"One-click process failed \n Exception: {str(e)}"})

class TrainRequest(BaseModel):
    data_dir: str  # 训练数据文件夹路径
    opt_dir: str  # 模型输出文件夹
    batch_size: int  # 批处理大小
    total_epoch: int  # 总训练轮数
    model_name: str  # 模型名称
    text_low_lr_rate: float  # 文本模块学习率权重
    if_save_latest: bool  # 是否保存最新模型
    if_save_every_weights: bool  # 是否保存每个权重
    save_every_epoch: int  # 每多少轮保存一次
    gpu_numbers: str  # GPU编号
    pretrained_s2G: str  # 预训练S2G模型路径
    pretrained_s2D: str  # 预训练S2D模型路径
    pretrained_s1: str  # 预训练S1模型路径
    notice_url: str  # 消息发送URL

@APP.post("/train_sovits")
async def train_sovits(request: TrainRequest):
    try:
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = json.loads(f.read())
        s2_dir = my_utils.clean_path(request.opt_dir)
        os.makedirs(f"{s2_dir}/logs_s2", exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if not is_half:
            data["train"]["fp16_run"] = False
            request.batch_size = max(1, request.batch_size // 2)
        data["train"]["batch_size"] = request.batch_size
        data["train"]["epochs"] = request.total_epoch
        data["train"]["text_low_lr_rate"] = request.text_low_lr_rate
        data["train"]["pretrained_s2G"] = request.pretrained_s2G
        data["train"]["pretrained_s2D"] = request.pretrained_s2D
        data["train"]["if_save_latest"] = request.if_save_latest
        data["train"]["if_save_every_weights"] = request.if_save_every_weights
        data["train"]["save_every_epoch"] = request.save_every_epoch
        data["train"]["gpu_numbers"] = request.gpu_numbers
        data["model"]["version"] = version
        data["data"]["exp_dir"] = s2_dir
        data["save_weight_dir"] = os.path.join(s2_dir, "SoVITS_weights")
        data["name"] = request.model_name
        data["version"] = version
        tmp_config_path = f"{tmp}/tmp_s2.json"
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        cmd = f'"{python_exec}" GPT_SoVITS/s2_train.py --config "{tmp_config_path}"'
        Popen(cmd, shell=True)

        # TODO: taskId 
        return JSONResponse(status_code=200, content={"message": "SoVITS training started", "taskId": "train_sovits", "opt_file": os.path.join(s2_dir, "SoVITS_weights")})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"SoVITS training failed\n Exception: {str(e)}"})

@APP.post("/train_gpt")
async def train_gpt(request: TrainRequest):
    try:
        config_file = "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        with open(config_file) as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        s1_dir = my_utils.clean_path(request.opt_dir)
        os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if not is_half:
            data["train"]["precision"] = "32"
            request.batch_size = max(1, request.batch_size // 2)
        data["train"]["batch_size"] = request.batch_size
        data["train"]["epochs"] = request.total_epoch
        data["pretrained_s1"] = request.pretrained_s1
        data["train"]["save_every_n_epoch"] = request.save_every_epoch
        data["train"]["if_save_every_weights"] = request.if_save_every_weights
        data["train"]["if_save_latest"] = request.if_save_latest
        data["train"]["if_dpo"] = False
        data["train"]["half_weights_save_dir"] = GPT_weight_root[-int(version[-1]) + 2]
        data["train"]["exp_name"] = s1_dir
        data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
        data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
        data["output_dir"] = f"{s1_dir}/logs_s1"
        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(request.gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = f"{tmp}/tmp_s1.yaml"
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        
        cmd = f'"{python_exec}" GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
        Popen(cmd, shell=True)
        
        return JSONResponse(status_code=200, content={"message": "GPT training started", "taskId": "train_gpt", "opt_file": os.path.join(s1_dir, "GPT_weights")})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "GPT training failed", "Exception": str(e)})

if __name__ == "__main__":
    try:
        if host == 'None':   # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input

def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs

# 添加缺失的变量
set_gpu_numbers = {0}  # 示例值，根据实际情况调整
default_gpu_numbers = "0"  # 示例值，根据实际情况调整
tmp = os.path.join(now_dir, "TEMP")  # 临时目录路径

"""
# API 文档

## `POST /uvr`
**描述**: 执行 UVR 转换任务。

**请求体**:

```json
{
  "inp_dir": "string", // 输入文件夹路径
  "model_name": "string",  // UVR模型名称，可选：onnx_dereverb_By_FoxJoy, HP5_only_main_vocal, HP3_all_vocals, HP2_all_vocals, VR-DeEchoNormal, VR-DeEchoAggressive, VR-DeEchoDeReverb
  "opt_dir_vocal": "string",  // 保存人声文件夹路径
  "opt_dir_ins": "string",  // 保存伴奏文件夹路径
  "agg": 0,  // （可选）人声提取激进程度，0-20的整数
  "format0": "string"  // 文件格式。可选："wav", "flac", "mp3", "m4a"
}
```

**响应**:

+ 成功: 返回 `200` 状态码和输出文件夹路径。

```json
{
  "opt_dir_ins": "string",
  "opt_dir_vocal": "string"
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /slice_audio`
**描述**: 执行音频切割任务。

**请求体**:

```json
{
  "inp_dir": "string", // 输入文件夹路径
  "opt_dir": "string",  // 输出文件夹路径
  "threshold": -34,  // 阈值（db），音量小于这个值视作静音的备选切割点
  "min_length": 4000,  // 最小长度（ms），每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
  "min_interval": 300,  // 最小间隔（ms），最短切割间隔
  "hop_size": 10,  // 跳跃大小，决定怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
  "max_sil_kept": 30,  // 最大保留静音（ms），切完后静音最多留多长
  "_max": 0.9,  // 最大值（0.0-1.0），归一化后最大值多少
  "alpha": 0.25,  // 阿尔法值（0.0-1.0），混多少比例归一化后音频进来
  "n_process": 4  // 使用进程数（1-12）
}
```

**响应**:

+ 成功: 返回 `200` 状态码和输出文件夹路径。

```json
{
  "opt_dir": "string"
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /denoise_audio`
**描述**: 执行语音降噪任务。

**请求体**:

```json
{
  "inp_dir": "string",  // 输入文件夹路径
  "opt_dir": "string"  // 输出文件夹路径
}
```

**响应**:

+ 成功: 返回 `200` 状态码和输出文件夹路径。

```json
{
  "opt_dir": "string"
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /asr`
**描述**: 执行自动语音识别 (ASR) 任务。

**请求体**:

```json
{
  "inp_dir": "string",  // 输入文件夹路径
  "opt_dir": "string",  // 输出文件夹路径
  "model": "string",  // 模型名称，可选：“达摩 ASR (中文)”，“Faster Whisper (多语种)”
  "model_size": "string",  // 模型大小
  "lang": "string",  // 语言
  "precision": "string"  // 精度
}

不同模型对应lang，precision和model_size的可选值：
"达摩 ASR (中文)": {
    'lang': ['zh','yue'],
    'size': ['large'],
    'precision': ['float32']
},
"Faster Whisper (多语种)": {
    'lang': ['auto', 'zh', 'en', 'ja', 'ko', 'yue'],
    'size': [
        "tiny",     "tiny.en", 
        "base",     "base.en", 
        "small",    "small.en", 
        "medium",   "medium.en", 
        "large",    "large-v1", 
        "large-v2", "large-v3"],
    'precision': ['float32', 'float16', 'int8']
},
```

**响应**:

+ 成功: 返回 `200` 状态码和输出文件夹路径。

```json
{
  "opt_dir": "string", // 输出文件夹路径
  "opt_text_dir": "string" // asr识别文本文件输出路径
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /one_click`
**描述**: 执行一键三连任务。

**请求体**:

```json
{
  "inp_text": "string",  // 文本标注的路径，参考asr任务的输出
  "inp_dir": "string",  // 输入音频文件夹路径，参考asr任务的输出
  "opt_dir": "string", // 输出结果文件夹
  "gpu_numbers1a": "string",  // 文本获取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
  "gpu_numbers1Ba": "string",  // SSL自监督特征提取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
  "gpu_numbers1c": "string",  // 语义token提取任务的GPU编号，GPU卡号以-分割，每个卡号一个进程，默认0-0
  "bert_pretrained_dir": "string",  // 预训练的中文BERT模型路径，默认为“GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large”
  "ssl_pretrained_dir": "string",  // 预训练的SSL模型路径，默认为“GPT_SoVITS/pretrained_models/chinese-hubert-base”
  "pretrained_s2G_path": "string"  // 预训练的SoVITS-G模型路径，默认为“GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth”
}
```

**响应**:

+ 成功: 返回 `200` 状态码和输出文件夹路径。

```json
{
  "opt_dir": "string", // 输出文件夹路径
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /train_sovits`
**描述**: 执行 SoVITS 模型训练任务。

**请求体**:

```json
{
  "data_dir": "string", // 训练数据文件夹路径。参考一键三连输出文件夹
  "opt_dir": "string", // 模型输出文件夹
  "batch_size": 0,  // 批处理大小
  "total_epoch": 0,  // 总训练轮数
  "model_name": "string",  // 模型名称
  "text_low_lr_rate": 0.0,  // 文本模块学习率权重
  "if_save_latest": true,  // 是否保存最新模型
  "if_save_every_weights": true,  // 是否保存每个权重
  "save_every_epoch": 0,  // 每多少轮保存一次
  "gpu_numbers": "string",  // GPU编号
  "pretrained_s2G": "string",  // 预训练S2G模型路径，默认：“GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth”
  "pretrained_s2D": "string",  // 预训练S2D模型路径，默认：“GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth”
  "pretrained_s1": "string",  // 预训练的GPT模型路径，默认：“GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt”
  "notice_url": "string" // 消息发送URL
}
```

**响应**:

+ 成功: 返回 `200` 状态码、任务id和模型输出路径。

```json
{
  "taskId": "string", // 训练任务ID
  "opt_file": "string", // 模型输出路径
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /train_gpt`
**描述**: 执行 GPT 模型训练任务。

**请求体**:

```json
{
  "data_dir": "string", // 训练数据文件夹路径。参考一键三连输出文件夹
  "opt_dir": "string", // 模型输出文件夹
  "batch_size": 0,  // 批处理大小
  "total_epoch": 0,  // 总训练轮数
  "model_name": "string",  // 模型名称
  "text_low_lr_rate": 0.0,  // 文本模块学习率权重
  "if_save_latest": true,  // 是否保存最新模型
  "if_save_every_weights": true,  // 是否保存每个权重
  "save_every_epoch": 0,  // 每多少轮保存一次
  "gpu_numbers": "string",  // GPU编号
  "pretrained_s2G": "string",  // 预训练S2G模型路径
  "pretrained_s2D": "string",  // 预训练S2D模型路径
  "pretrained_s1": "string",  // 预训练S1模型路径
  "notice_url": "string" // 消息发送URL
}
```

**响应**:

+ 成功: 返回 `200` 状态码、任务id和模型输出路径。

```json
{
  "taskId": "string", // 训练任务ID
  "opt_file": "string", // 模型输出路径
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `POST /delete_task`
**描述**: 终止或删除特定任务。

**请求体**:

```json
{
  "taskId": "string" // 任务id
}
```

**响应**:

+ 成功: 返回 `200` 状态码、任务id和模型输出路径。

```json
{
  "taskId": "string", // 训练任务ID
  "log": "string" // 执行结果
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```

## `GET /get_task_info`
**描述**: 查询训练任务状态。

**请求体**:

```json
{
  "taskId": "string" // 任务ID
}
```

**响应**:

+ 成功: 返回 `200` 状态码、任务id和模型输出路径。

```json
{
  "taskId": "string", // 训练任务ID
  "status": "string", // 任务状态。可选值：
                      // 	“FAIL”：任务失败，
                      // 	”PENDING“：等待开始
                      // 	“SUCCESS”：任务成功
                      // 	“RUNNING”：任务进行中
  "log": "string"     // 执行log
}
```

+ 失败: 返回 `400` 状态码和错误信息。

```json
{
  "error": "string"
}
```


"""