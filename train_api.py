"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
    "sovits_weights_path": ""     # str.(optional) path to the sovits weights file.
    "gpt_weights_path": ""        # str.(optional) path to the gpt weights file.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
    
"""
import os
import sys
import traceback
from typing import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
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

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()
class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k:int = 5
    top_p:float = 1
    temperature:float = 1
    text_split_method:str = "cut5"
    batch_size:int = 1
    batch_threshold:float = 0.75
    split_bucket:bool = True
    speed_factor:float = 1.0
    fragment_interval:float = 0.3
    seed:int = -1
    media_type:str = "wav"
    streaming_mode:bool = False
    parallel_infer:bool = True
    repetition_penalty:float = 1.35

### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if (media_type == "ogg"):
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif (media_type == "aac"):
        io_buffer = pack_aac(io_buffer, data, rate)
    elif (media_type == "wav"):
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer



# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command:str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req:dict):
    text:str = req.get("text", "")
    text_lang:str = req.get("text_lang", "")
    ref_audio_path:str = req.get("ref_audio_path", "")
    streaming_mode:bool = req.get("streaming_mode", False)
    media_type:str = req.get("media_type", "wav")
    prompt_lang:str = req.get("prompt_lang", "")
    text_split_method:str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if (text_lang in [None, ""]) :
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"})
    if (prompt_lang in [None, ""]) :
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and  not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
    
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})

    return None

async def tts_handle(req:dict):
    """
    Text to speech handler.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.  
                "sovits_weights_path": ""     # str.(optional) path to the sovits weights file.
                "gpt_weights_path": ""        # str.(optional) path to the gpt weights file.        
            }
    returns:
        StreamingResponse: audio stream response.
    """
    
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")
    sovits_weights_path = req.get("sovits_weights_path", None)
    gpt_weights_path = req.get("gpt_weights_path", None)

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    if sovits_weights_path is not None:
        set_sovits_weights(sovits_weights_path)
    
    if gpt_weights_path is not None:
        set_gpt_weights(gpt_weights_path)
    
    try:
        tts_generator=tts_pipeline.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator:Generator, media_type:str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")
    
        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})
    





@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)



@APP.get("/tts")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = None,
                        ref_audio_path: str = None,
                        aux_ref_audio_paths:list = None,
                        prompt_lang: str = None,
                        prompt_text: str = "",
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut0",
                        batch_size:int = 1,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1.0,
                        fragment_interval:float = 0.3,
                        seed:int = -1,
                        media_type:str = "wav",
                        streaming_mode:bool = False,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.35
                        ):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle(req)
                

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})
        
#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())
            
#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})

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
    model_name: str  # 模型名称
    inp_root: str  # 输入文件夹路径
    save_root_vocal: str  # 保存人声文件夹路径
    paths: list  # 文件路径列表
    save_root_ins: str  # 保存伴奏文件夹路径
    agg: int  # 聚合参数
    format0: str  # 文件格式
    device: str  # 设备
    is_half: bool

@APP.post("/uvr_convert")
async def uvr_convert(request: UVRRequest):
    request = request.dict()
    try:
        model_name = request.model_name
        inp_root = request.inp_root
        save_root_vocal = request.save_root_vocal
        paths = request.paths
        save_root_ins = request.save_root_ins
        agg = request.agg
        format0 = request.format0
        device = request.device
        is_half = request.is_half

        if model_name not in uvr5_names:
            return JSONResponse(status_code=400, content={"message": f"model_name: {model_name} is not supported"})

        response = []
        async for info in uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, device, is_half):
            response.append(info)
        
        return JSONResponse(status_code=200, content={"message": "success", "details": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "uvr_convert failed", "Exception": str(e)})

class SliceRequest(BaseModel):
    threshold: str  # 阈值
    min_length: str  # 最小长度
    min_interval: str  # 最小间隔
    hop_size: str  # 跳跃大小
    max_sil_kept: str  # 最大保留静音
    _max: float  # 最大值
    alpha: float  # 阿尔法值
    n_parts: int  # 分片数量

@APP.post("/slice_audio")
async def slice_audio(request: SliceRequest, audio_file: UploadFile = File(...)):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
            
            # 检查文件是否存在
            if not os.path.exists(temp_file_path):
                return JSONResponse(status_code=400, content={"message": "Input file does not exist"})
            
            # 创建输出文件夹
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            ps_slice = []
            for i_part in range(request.n_parts):
                cmd = f'"{python_exec}" tools/slice_audio.py "{temp_file_path}" "{output_dir}" {request.threshold} {request.min_length} {request.min_interval} {request.hop_size} {request.max_sil_kept} {request._max} {request.alpha} {i_part} {request.n_parts}'
                p = Popen(cmd, shell=True)
                ps_slice.append(p)
            
            for p in ps_slice:
                p.wait()
            
            # 假设处理后的文件名为 output.wav
            output_file_path = os.path.join(output_dir, "output.wav")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"message": "Output file does not exist"})
            
            return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Slicing failed", "Exception": str(e)})

class DenoiseRequest(BaseModel):
    pass  # 不再需要输入文件夹路径

@APP.post("/denoise_audio")
async def denoise_audio(request: DenoiseRequest, audio_file: UploadFile = File(...)):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
            
            # 检查文件是否存在
            if not os.path.exists(temp_file_path):
                return JSONResponse(status_code=400, content={"message": "Input file does not exist"})
            
            # 创建输出文件夹
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建命令
            cmd = f'"{python_exec}" tools/cmd-denoise.py -i "{temp_file_path}" -o "{output_dir}" -p {"float16" if is_half else "float32"}'
            p = Popen(cmd, shell=True)
            p.wait()
            
            # 假设处理后的文件名为 output.wav
            output_file_path = os.path.join(output_dir, "output.wav")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"message": "Output file does not exist"})
            
            return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Denoising failed", "Exception": str(e)})

class ASRRequest(BaseModel):
    model: str  # 模型名称
    model_size: str  # 模型大小
    lang: str  # 语言
    precision: str  # 精度

@APP.post("/asr")
async def asr(request: ASRRequest, audio_file: UploadFile = File(...)):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
            
            # 检查文件是否存在
            if not os.path.exists(temp_file_path):
                return JSONResponse(status_code=400, content={"message": "Input file does not exist"})
            
            # 创建输出文件夹
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建命令
            cmd = f'"{python_exec}" tools/asr/{asr_dict[request.model]["path"]} -i "{temp_file_path}" -o "{output_dir}" -s {request.model_size} -l {request.lang} -p {request.precision}'
            p = Popen(cmd, shell=True)
            p.wait()
            
            # 假设处理后的文件名为 output.txt
            output_file_path = os.path.join(output_dir, "output.txt")
            if not os.path.exists(output_file_path):
                return JSONResponse(status_code=400, content={"message": "Output file does not exist"})
            
            return FileResponse(output_file_path, media_type="text/plain", filename="output.txt")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "ASR task failed", "Exception": str(e)})

class OneClickRequest(BaseModel):
    inp_text: str  # 输入文本路径
    exp_name: str  # 实验名称
    gpu_numbers1a: str  # GPU编号1a
    gpu_numbers1Ba: str  # GPU编号1Ba
    gpu_numbers1c: str  # GPU编号1c
    bert_pretrained_dir: str  # BERT预训练模型路径
    ssl_pretrained_dir: str  # SSL预训练模型路径
    pretrained_s2G_path: str  # 预训练S2G模型路径

@APP.post("/one_click")
async def one_click(request: OneClickRequest, audio_file: UploadFile = File(...)):
    try:
        # 创建临时目录保存上传的文件和输出文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, audio_file.filename)
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
            
            # 检查文件是否存在
            if not os.path.exists(temp_file_path):
                return JSONResponse(status_code=400, content={"message": "Input file does not exist"})
            
            # 创建输出文件夹
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输入文件夹路径为临时目录
            inp_wav_dir = temp_dir
            
            # 处理逻辑
            inp_text = my_utils.clean_path(request.inp_text)
            if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
                check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
            opt_dir = f"{output_dir}/{request.exp_name}"
            os.makedirs(opt_dir, exist_ok=True)
            
            ps1abc = []
            # 1a
            path_text = f"{opt_dir}/2-name2text.txt"
            if not os.path.exists(path_text) or (os.path.exists(path_text) and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": request.exp_name,
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
                "inp_wav_dir": inp_wav_dir,
                "exp_name": request.exp_name,
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
                    "exp_name": request.exp_name,
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
            return JSONResponse(status_code=200, content={"message": "One-click process completed"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "One-click process failed", "Exception": str(e)})

class TrainRequest(BaseModel):
    batch_size: int  # 批处理大小
    total_epoch: int  # 总训练轮数
    exp_name: str  # 实验名称
    text_low_lr_rate: float  # 文本低学习率
    if_save_latest: bool  # 是否保存最新模型
    if_save_every_weights: bool  # 是否保存每个权重
    save_every_epoch: int  # 每多少轮保存一次
    gpu_numbers: str  # GPU编号
    pretrained_s2G: str = None  # 预训练S2G模型路径
    pretrained_s2D: str = None  # 预训练S2D模型路径
    pretrained_s1: str = None  # 预训练S1模型路径
    if_dpo: bool = False  # 是否使用DPO

@APP.post("/train_sovits")
async def train_sovits(request: TrainRequest):
    try:
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = json.loads(f.read())
        s2_dir = f"{exp_root}/{request.exp_name}"
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
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root[-int(version[-1]) + 2]
        data["name"] = request.exp_name
        data["version"] = version
        tmp_config_path = f"{tmp}/tmp_s2.json"
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        
        await request.send(JSONResponse(status_code=200, content={"message": "SoVITS training started"}))

        cmd = f'"{python_exec}" GPT_SoVITS/s2_train.py --config "{tmp_config_path}"'
        p = Popen(cmd, shell=True)
        p.wait()

        return JSONResponse(status_code=200, content={"message": "SoVITS training completed"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "SoVITS training failed", "Exception": str(e)})

@APP.post("/train_gpt")
async def train_gpt(request: TrainRequest):
    try:
        config_file = "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        with open(config_file) as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        s1_dir = f"{exp_root}/{request.exp_name}"
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
        data["train"]["if_dpo"] = request.if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root[-int(version[-1]) + 2]
        data["train"]["exp_name"] = request.exp_name
        data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
        data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
        data["output_dir"] = f"{s1_dir}/logs_s1"
        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(request.gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = f"{tmp}/tmp_s1.yaml"
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))

        await request.send(JSONResponse(status_code=200, content={"message": "GPT training started"}))
        
        cmd = f'"{python_exec}" GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
        p = Popen(cmd, shell=True)
        p.wait()
        
        return JSONResponse(status_code=200, content={"message": "GPT training completed"})
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

## `POST /uvr_convert`

**描述**: 执行 UVR 转换任务。

**请求体**:
```json
{
  "model_name": "string",  // 模型名称
  "inp_root": "string",  // 输入文件夹路径
  "save_root_vocal": "string",  // 保存人声文件夹路径
  "paths": ["string"],  // 文件路径列表
  "save_root_ins": "string",  // 保存伴奏文件夹路径
  "agg": 0,  // 聚合参数
  "format0": "string"  // 文件格式
}
```

**响应**:
- 成功: 返回 `200` 状态码和转换详情。
- 失败: 返回 `400` 状态码和错误信息。

## `POST /slice_audio`

**描述**: 执行音频切割任务。

**请求体**:
```json
{
  "inp": "string",  // 输入文件路径
  "opt_root": "string",  // 输出文件夹路径
  "threshold": "string",  // 阈值
  "min_length": "string",  // 最小长度
  "min_interval": "string",  // 最小间隔
  "hop_size": "string",  // 跳跃大小
  "max_sil_kept": "string",  // 最大保留静音
  "_max": 0.0,  // 最大值
  "alpha": 0.0,  // 阿尔法值
  "n_parts": 0  // 分片数量
}
```

**响应**:
- 成功: 返回 `200` 状态码和输出路径。
- 失败: 返回 `400` 状态码和错误信息。

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
- 成功: 返回 `200` 状态码和输出路径。
- 失败: 返回 `400` 状态码和错误信息。

## `POST /asr`

**描述**: 执行自动语音识别 (ASR) 任务。

**请求体**:
```json
{
  "inp_dir": "string",  // 输入文件夹路径
  "opt_dir": "string",  // 输出文件夹路径
  "model": "string",  // 模型名称
  "model_size": "string",  // 模型大小
  "lang": "string",  // 语言
  "precision": "string"  // 精度
}
```

**响应**:
- 成功: 返回 `200` 状态码和输出路径。
- 失败: 返回 `400` 状态码和错误信息。

## `POST /one_click`

**描述**: 执行一键三连任务。

**请求体**:
```json
{
  "inp_text": "string",  // 输入文本路径
  "inp_wav_dir": "string",  // 输入音频文件夹路径
  "exp_name": "string",  // 实验名称
  "gpu_numbers1a": "string",  // GPU编号1a
  "gpu_numbers1Ba": "string",  // GPU编号1Ba
  "gpu_numbers1c": "string",  // GPU编号1c
  "bert_pretrained_dir": "string",  // BERT预训练模型路径
  "ssl_pretrained_dir": "string",  // SSL预训练模型路径
  "pretrained_s2G_path": "string"  // 预训练S2G模型路径
}
```

**响应**:
- 成功: 返回 `200` 状态码和成功消息。
- 失败: 返回 `400` 状态码和错误信息。

## `POST /train_sovits`

**描述**: 执行 SoVITS 模型训练任务。

**请求体**:
```json
{
  "batch_size": 0,  // 批处理大小
  "total_epoch": 0,  // 总训练轮数
  "exp_name": "string",  // 实验名称
  "text_low_lr_rate": 0.0,  // 文本低学习率
  "if_save_latest": true,  // 是否保存最新模型
  "if_save_every_weights": true,  // 是否保存每个权重
  "save_every_epoch": 0,  // 每多少轮保存一次
  "gpu_numbers": "string",  // GPU编号
  "pretrained_s2G": "string",  // 预训练S2G模型路径
  "pretrained_s2D": "string",  // 预训练S2D模型路径
  "pretrained_s1": "string",  // 预训练S1模型路径
  "if_dpo": false  // 是否使用DPO
}
```

**响应**:
- 成功: 返回 `200` 状态码和成功消息。
- 失败: 返回 `400` 状态码和错误信息。

## `POST /train_gpt`

**描述**: 执行 GPT 模型训练任务。

**请求体**:
```json
{
  "batch_size": 0,  // 批处理大小
  "total_epoch": 0,  // 总训练轮数
  "exp_name": "string",  // 实验名称
  "text_low_lr_rate": 0.0,  // 文本低学习率
  "if_save_latest": true,  // 是否保存最新模型
  "if_save_every_weights": true,  // 是否保存每个权重
  "save_every_epoch": 0,  // 每多少轮保存一次
  "gpu_numbers": "string",  // GPU编号
  "pretrained_s2G": "string",  // 预训练S2G模型路径
  "pretrained_s2D": "string",  // 预训练S2D模型路径
  "pretrained_s1": "string",  // 预训练S1模型路径
  "if_dpo": false  // 是否使用DPO
}
```

**响应**:
- 成功: 返回 `200` 状态码和成功消息。
- 失败: 返回 `400` 状态码和错误信息。
"""