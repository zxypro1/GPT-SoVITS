import os,shutil,sys,pdb,re
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import json,yaml,warnings,torch
import platform
import psutil
import signal
from datetime import datetime
from tools.subfix_webui import *
from tools.uvr5.webui import *
from GPT_SoVITS.inference_webui import *

warnings.filterwarnings("ignore")
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if(os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if(name=="jieba.cache"):continue
        path="%s/%s"%(tmp,name)
        delete=os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass
from tools import my_utils
import traceback
import shutil
import pdb
import gradio as gr
from subprocess import Popen
import signal
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto(language='zh-CN')
from scipy.io import wavfile
from tools.my_utils import load_audio
from multiprocessing import cpu_count

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu

n_cpu=cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060"]):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    default_batch_size = psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2
gpus = "-".join([i[0] for i in gpu_infos])

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append(name)
    return SoVITS_names,GPT_names
SoVITS_weight_root=os.environ['download_path'] + "/SoVITS_weights"
GPT_weight_root=os.environ['download_path'] + "/GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
SoVITS_names,GPT_names = get_weights_names()

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

p_label=None
p_uvr5=None
p_asr=None
p_denoise=None
p_tts_inference=None

def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def change_label(path_list, index_slider, batchsize_slider):
    path_list=str(my_utils.clean_path(path_list))
    set_global(load_json="None",
               load_list=path_list,
               json_key_path="wav_path",
               json_key_text="text",
               batch=10)
    return b_change_index(index_slider, batchsize_slider)

def change_uvr5(if_uvr5):
    global p_uvr5
    if(if_uvr5==True and p_uvr5==None):
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s %s'%(python_exec,infer_device,is_half,webui_port_uvr5,is_share)
        yield i18n("UVR5已开启")
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    elif(if_uvr5==False and p_uvr5!=None):
        kill_process(p_uvr5.pid)
        p_uvr5=None
        yield i18n("UVR5已关闭")

def change_tts_inference(bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path):
    os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
    os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
    os.environ["cnhubert_base_path"]=cnhubert_base_path
    os.environ["bert_path"]=bert_path
    os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
    os.environ["is_half"]=str(is_half)
    os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
    os.environ["is_share"]=str(is_share)

def change_tts_mode(tts_mode):
    if tts_mode == "模版音频":
        return {"__type__":"update", "visible":False, "value": None}, {"__type__":"update", "visible":False, "value": None}, {"__type__":"update", "visible":False}, {"__type__":"update", "visible":True, "value": None}, {"__type__":"update", "visible":True, "value": None}
    else:
        return {"__type__":"update", "visible":True, "value": None}, {"__type__":"update", "visible":True, "value": None}, {"__type__":"update", "visible":True}, {"__type__":"update", "visible":False, "value": None}, {"__type__":"update", "visible":False, "value": None}

template_audio_path = {
    # i18n("红色女恶魔"): "template_audio/female_demon1.wav",
    # i18n("白色女恶魔"): "template_audio/female_demon2.wav",
    # i18n("男恶魔"): "template_audio/male_demon.wav",
    # i18n("高等精灵"): "template_audio/high_elf.wav",
    i18n("小精灵"): "template_audio/small_elf.wav",
    i18n("甜美女声"): "template_audio/serverless_goddess.wav",
}

template_audio_text = {
    # i18n("红色女恶魔"): "《三国演义》由东汉末年黄巾起义末期开始描写，至西晋初期国家重归统一结束。",
    # i18n("白色女恶魔"): "《三国演义》由东汉末年黄巾起义末期开始描写，至西晋初期国家重归统一结束。",
    # i18n("男恶魔"): "《三国演义》由东汉末年黄巾起义末期开始描写，至西晋初期国家重归统一结束。",
    # i18n("高等精灵"): "《三国演义》由东汉末年黄巾起义末期开始描写，至西晋初期国家重归统一结束。",
    i18n("小精灵"): "《三国演义》由东汉末年黄巾起义末期开始描写，至西晋初期国家重归统一结束。",
    i18n("甜美女声"): "函数计算是事件驱动的全托管计算服务。通过函数计算，您无需管理服务器等基础设施。",
}

template_audio_image = {
    # i18n("红色女恶魔"): "template_images/red_demon.png",
    # i18n("白色女恶魔"): "template_images/white_demon.png",
    # i18n("男恶魔"): "template_images/demon.png",
    # i18n("高等精灵"): "template_images/high_elf.png",
    i18n("小精灵"): "template_images/small_elf.png",
    i18n("甜美女声"): "template_images/serverless_goddess.png",
}

def change_template_text(template_text):
    if template_text in template_audio_path:
        return {"__type__": "update", "value": template_audio_text[template_text]}, {"__type__": "update", "value": template_audio_path[template_text]}, {"__type__": "update", "value": template_audio_image[template_text]}
    else:
        return {"__type__": "update", "value": None}, {"__type__": "update", "value": None}, {"__type__": "update", "value": None}


from tools.asr.config import asr_dict
def open_asr(
        model_choose,
        dir_wav_input,
        wav_inputs,
        agg,
        format0,

        asr_model,
        asr_model_size,
        asr_lang,

        threshold,
        min_length,
        min_interval,
        hop_size,
        max_sil_kept,
        _max,
        alpha,
        n_process,

        opt_vocal_root = os.environ['download_path'] + "/output/uvr5_opt",
        opt_ins_root = os.environ['download_path'] + "/output/uvr5_opt",
        slice_inp_path = os.environ['download_path'] + "/output/uvr5_opt",
        slice_opt_root = os.environ['download_path'] + "/output/slicer_opt",
        denoise_input_dir = os.environ['download_path'] + "/output/slicer_opt",
        denoise_output_dir = os.environ['download_path'] + "/output/denoise_opt",
        asr_inp_dir = os.environ['download_path'] + "/output/denoise_opt",
        asr_opt_dir = os.environ['download_path'] + "/output/asr_opt",
):
    for i in uvr(
        model_choose,
        dir_wav_input,
        opt_vocal_root,
        wav_inputs,
        opt_ins_root,
        agg,
        format0,
    ):
        yield i, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}
    for i in open_slice(
        slice_inp_path,
        slice_opt_root,
        threshold,
        min_length,
        min_interval,
        hop_size,
        max_sil_kept,
        _max,
        alpha,
        n_process
    ):
        yield i

    for i in open_denoise(
        denoise_input_dir,
        denoise_output_dir,
    ):
        yield i
    global p_asr
    if(p_asr==None):
        asr_inp_dir=my_utils.clean_path(asr_inp_dir)
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s"%("float16"if is_half==True else "float32")

        yield "ASR任务开启：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr=None
        yield f"ASR任务完成, 查看终端进行下一步",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的ASR任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        # return None

def close_asr():
    for i in close_slice():
        yield i
    for i in close_denoise():
        yield i
    global p_asr
    if(p_asr!=None):
        kill_process(p_asr.pid)
        p_asr=None
    return "已终止ASR进程",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if(p_denoise==None):
        denoise_inp_dir=my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir=my_utils.clean_path(denoise_opt_dir)
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s'%(python_exec,denoise_inp_dir,denoise_opt_dir,"float16"if is_half==True else "float32")

        yield "语音降噪任务开启：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise=None
        yield f"语音降噪任务完成, 查看终端进行下一步",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的语音降噪任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        # return None

def close_denoise():
    global p_denoise
    if(p_denoise!=None):
        kill_process(p_denoise.pid)
        p_denoise=None
    return "已终止语音降噪进程",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_SoVITS=None
def open1Ba(
        batch_size,
        total_epoch,
        exp_name,
        text_low_lr_rate,
        if_save_latest,
        if_save_every_weights,
        save_every_epoch,
        gpu_numbers1Ba,
        pretrained_s2G,
        pretrained_s2D,

        inp_text,
        inp_wav_dir,
        bert_pretrained_dir,
        cnhubert_base_dir,
):
    global p_train_SoVITS
    for update in open1abc(
        inp_text=inp_text,
        inp_wav_dir=inp_wav_dir,
        bert_pretrained_dir=bert_pretrained_dir,
        ssl_pretrained_dir=cnhubert_base_dir,
        exp_name=exp_name,
        gpu_numbers1Ba=gpu_numbers1Ba,
        gpu_numbers1a=gpu_numbers1Ba,
        gpu_numbers1c=gpu_numbers1Ba,
        pretrained_s2G_path=pretrained_s2G,
    ):
        yield update
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root
        data["name"]=exp_name
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        yield "SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        yield "SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Ba():
    for i in close1abc():
        yield i
    global p_train_SoVITS
    if(p_train_SoVITS!=None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS=None
    return "已终止SoVITS训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_GPT=None
def open1Bb(
        batch_size,
        total_epoch,
        exp_name,
        if_dpo,
        if_save_latest,
        if_save_every_weights,
        save_every_epoch,
        gpu_numbers,
        pretrained_s1,

        pretrained_s2G,
        inp_text,
        inp_wav_dir,
        bert_pretrained_dir,
        cnhubert_base_dir,
):
    global p_train_GPT

    for update in open1abc(
        inp_text=inp_text,
        inp_wav_dir=inp_wav_dir,
        bert_pretrained_dir=bert_pretrained_dir,
        ssl_pretrained_dir=cnhubert_base_dir,
        exp_name=exp_name,
        gpu_numbers1Ba=gpu_numbers,
        gpu_numbers1a=gpu_numbers,
        gpu_numbers1c=gpu_numbers,
        pretrained_s2G_path=pretrained_s2G,
    ):
        yield update
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        yield "GPT训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        yield "GPT训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Bb():
    for i in close1abc():
        yield i
    global p_train_GPT
    if(p_train_GPT!=None):
        kill_process(p_train_GPT.pid)
        p_train_GPT=None
    return "已终止GPT训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

ps_slice=[]
def open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    if(os.path.exists(inp)==False):
        yield "输入路径不存在",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if os.path.isfile(inp):n_parts=1
    elif os.path.isdir(inp):pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (python_exec,inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps_slice:
            p.wait()
        ps_slice=[]
        yield "切割结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice=[]
    return "已终止所有切割进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1a=[]
def open1a(inp_text,inp_wav_dir,exp_name,gpu_numbers,bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1a == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "bert_pretrained_dir":bert_pretrained_dir,
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield "文本进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a=[]
        yield "文本进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的文本任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1a():
    global ps1a
    if (ps1a != []):
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a=[]
    return "已终止所有1a进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1b=[]
def open1b(inp_text,inp_wav_dir,exp_name,gpu_numbers,ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1b == []):
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":"%s/%s"%(exp_root,exp_name),
            "cnhubert_base_dir":ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "SSL提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1b:
            p.wait()
        ps1b=[]
        yield "SSL提取进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SSL提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b=[]
    return "已终止所有1b进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1c=[]
def open1c(inp_text,exp_name,gpu_numbers,pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if (ps1c == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "pretrained_s2G":pretrained_s2G_path,
            "s2config_path":"GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "语义token提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c=[]
        yield "语义token提取进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的语义token提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c=[]
    return "已终止所有语义token进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc=[]
def open1abc(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            for p in ps1abc:p.wait()
            yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

#### 新方法
def change_inp_ref(inp_ref):
    if inp_ref == "" or inp_ref == None: return
    output_dir = os.environ["download_path"] + '/upload_files/'
    os.makedirs(output_dir, exist_ok=True)
    # 获取当前时间戳

    # 获取上传的文件名
    base_name = os.path.splitext(os.path.basename(inp_ref))[0]

    # 生成新的文件名
    new_filename = f"{base_name}.wav"
    output_path = os.path.join(output_dir, new_filename)

    # 将上传的音频文件保存到新的文件名
    shutil.copy(inp_ref, output_path)
    print(f"上传音频文件已保存到: {output_path}")

def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    gr.Markdown(
        value=
            i18n("中文教程文档：https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e")
    )

    with gr.Tabs():
        with gr.TabItem("0-语音克隆 && 推理"):
            gr.Markdown(value="## " + i18n(
                "语音克隆 && 推理"))
            gr.Markdown(value=i18n(
                "欢迎使用FC版GPT-SoVITS。<br>快速开始：<br>1. 上传你的语音文件作为参考音频<br>2. 输入这段参考音频的文本和语言<br>3. 输入想要输出的文本<br>4. 选择输出文本的语言<br>点击合成语音，成功克隆自己的声音！"))
            # with gr.Row():
            #     # GPT_dropdown = gr.Dropdown(label=i18n("*GPT模型列表"), choices=sorted(GPT_names,key=custom_sort_key),value=pretrained_gpt_name,interactive=True)
            #     # SoVITS_dropdown = gr.Dropdown(label=i18n("*SoVITS模型列表"), choices=sorted(SoVITS_names,key=custom_sort_key),value=pretrained_sovits_name,interactive=True)
            #     gpu_number_1C = gr.Textbox(label=i18n("GPU卡号,只能填1个整数"), value=gpus, interactive=True)
            #     if_tts = gr.Button("刷新模型参数", variant="primary", visible=True)
            #     if_tts.click(change_tts_inference,
            #                  [bert_pretrained_dir, cnhubert_base_dir, gpu_number_1C, GPT_dropdown,
            #                   SoVITS_dropdown])
            with gr.Blocks(title="GPT-SoVITS WebUI"):
                with gr.Row():
                    GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"),
                                               choices=sorted(GPT_names, key=custom_sort_key),
                                               value=gpt_path,
                                               interactive=True)
                    SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"),
                                                  choices=sorted(SoVITS_names, key=custom_sort_key),
                                                  value=sovits_path, interactive=True)
                    refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                    refresh_button.click(fn=change_choices, inputs=[],
                                         outputs=[SoVITS_dropdown, GPT_dropdown])
                    SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
                    GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])
                gr.Markdown(value=i18n("*请上传并填写参考信息"))
                with gr.Row():
                    tts_mode = gr.Radio(["模版音频", "个人上传"], label="模式", info="使用模版音频还是自己上传？", value="模版音频")
                    inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath", visible=False, value=template_audio_path[i18n("小精灵")])
                    with gr.Column():
                        prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", visible=False)
                        prompt_language = gr.Dropdown(
                            label=i18n("参考音频的语种"),
                            choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"),
                                     i18n("日英混合"),
                                     i18n("多语种混合")], value=i18n("中文"),
                            visible=False
                        )
                        with gr.Row():
                            template_image = gr.Image(
                                show_label=False,
                                show_download_button=False,
                                value=template_audio_image[i18n("小精灵")]
                            )
                            template_text = gr.Dropdown(
                                label=i18n("选择默认语音模版"),
                                choices=template_audio_text.keys(),
                                value=i18n("小精灵")
                            )
                    with gr.Column():
                        text = gr.Textbox(label=i18n("需要生成的文本"), value="")
                        text_language = gr.Dropdown(
                            label=i18n("需要生成的语种"),
                            choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"),
                                     i18n("日英混合"),
                                     i18n("多语种混合")], value=i18n("中文")
                        )
                    inp_ref.change(change_inp_ref, inputs=[inp_ref])
                    tts_mode.change(change_tts_mode, inputs=[tts_mode], outputs=[inp_ref ,prompt_text, prompt_language, template_text, template_image])
                    template_text.change(change_template_text, inputs=[template_text], outputs=[prompt_text, inp_ref, template_image])
                with gr.Row():
                    inference_button = gr.Button(i18n("合成语音"), variant="primary")
                    output = gr.Audio(label=i18n("输出的语音"))
                gr.Markdown(value='## ' + i18n("参数设置"))
                with gr.Row():
                    how_to_cut = gr.Dropdown(
                        label=i18n("怎么切"),
                        choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"),
                                 i18n("按中文句号。切"),
                                 i18n("按英文句号.切"), i18n("按标点符号切"), ],
                        value=i18n("凑50字一切"),
                    )
                    with gr.Column():
                        ref_text_free = gr.Checkbox(
                            label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                            value=False, interactive=True, show_label=True)
                        gr.Markdown(i18n(
                            "使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。"))
                    with gr.Column():
                        gr.Markdown(value=i18n("gpt采样参数(无参考文本时不要太低)："))
                        top_k = gr.Slider(minimum=1, maximum=100, step=1, label=i18n("top_k"), value=5,
                                          interactive=True)
                        top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1,
                                          interactive=True)
                        temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("temperature"),
                                                value=1, interactive=True)
                inference_button.click(
                    get_tts_wav,
                    [inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p,
                     temperature, ref_text_free],
                    [output],
                )

        with gr.TabItem(i18n("1-数据预处理")):#提前随机切片防止uvr5爆内存->uvr5->slicer->asr->打标
            opt_vocal_root = os.environ['download_path'] + "/output/uvr5_opt"
            opt_ins_root = os.environ['download_path'] + "/output/uvr5_opt"
            slice_inp_path = os.environ['download_path'] + "/output/uvr5_opt"
            slice_opt_root = os.environ['download_path'] + "/output/slicer_opt"
            denoise_input_dir = os.environ['download_path'] + "/output/slicer_opt"
            denoise_output_dir = os.environ['download_path'] + "/output/denoise_opt"
            asr_inp_dir = os.environ['download_path'] + "/output/denoise_opt"
            asr_opt_dir = os.environ['download_path'] + "/output/asr_opt"
            with gr.Group():
                gr.Markdown(value="## " + i18n("语音预处理"))
                gr.Markdown(
                    value=i18n(
                        "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label=i18n("人声提取激进程度"),
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    # but2 = gr.Button(i18n("转换"), variant="primary")
                    # vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    # but2.click(
                    #     uvr,
                    #     [
                    #         model_choose,
                    #         dir_wav_input,
                    #         opt_vocal_root,
                    #         wav_inputs,
                    #         opt_ins_root,
                    #         agg,
                    #         format0,
                    #     ],
                    #     [vc_output4],
                    #     api_name="uvr_convert",
                    # )
            with gr.Row():
                open_asr_button = gr.Button(i18n("开启数据预处理"), variant="primary", visible=True)
                close_asr_button = gr.Button(i18n("终止数据预处理"), variant="primary", visible=False)
                asr_info = gr.Textbox(label=i18n("数据预处理输出信息"))

                def change_lang_choices(key): #根据选择的模型修改可选的语言
                    # return gr.Dropdown(choices=asr_dict[key]['lang'])
                    return {"__type__": "update", "choices": asr_dict[key]['lang'],"value":asr_dict[key]['lang'][0]}
                def change_size_choices(key): # 根据选择的模型修改可选的模型尺寸
                    # return gr.Dropdown(choices=asr_dict[key]['size'])
                    return {"__type__": "update", "choices": asr_dict[key]['size']}


            gr.Markdown(value="## " + i18n("参数设置"))
            gr.Markdown(value=i18n("语音切分参数"))
            with gr.Row():
                with gr.Row():
                    threshold=gr.Textbox(label=i18n("threshold:音量小于这个值视作静音的备选切割点"),value="-34")
                    min_length=gr.Textbox(label=i18n("min_length:每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值"),value="4000")
                    min_interval=gr.Textbox(label=i18n("min_interval:最短切割间隔"),value="300")
                    hop_size=gr.Textbox(label=i18n("hop_size:怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）"),value="10")
                    max_sil_kept=gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"),value="500")
                with gr.Column():
                    # open_slicer_button=gr.Button(i18n("开启语音切割"), variant="primary",visible=True)
                    # close_slicer_button=gr.Button(i18n("终止语音切割"), variant="primary",visible=False)
                    _max=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("max:归一化后最大值多少"),value=0.9,interactive=True)
                    alpha=gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("alpha_mix:混多少比例归一化后音频进来"),value=0.25,interactive=True)
                    n_process=gr.Slider(minimum=1,maximum=n_cpu,step=1,label=i18n("切割使用的进程数"),value=4,interactive=True)
            gr.Markdown(value=i18n("批量离线ASR工具参数"))
            with gr.Row():
                with gr.Column():
                    # with gr.Row():

                    with gr.Row():
                        asr_model = gr.Dropdown(
                            label       = i18n("ASR 模型"),
                            choices     = list(asr_dict.keys()),
                            interactive = True,
                            value="达摩 ASR (中文)"
                        )
                        asr_size = gr.Dropdown(
                            label       = i18n("ASR 模型尺寸"),
                            choices     = ["large"],
                            interactive = True,
                            value="large"
                        )
                        asr_lang = gr.Dropdown(
                            label       = i18n("ASR 语言设置"),
                            choices     = ["zh"],
                            interactive = True,
                            value="zh"
                        )


            open_asr_button.click(open_asr,
                                  [
                                      model_choose,
                                      dir_wav_input,
                                      wav_inputs,
                                      agg,
                                      format0,

                                      asr_model,
                                      asr_size,
                                      asr_lang,

                                      threshold,
                                      min_length,
                                      min_interval,
                                      hop_size,
                                      max_sil_kept,
                                      _max,
                                      alpha,
                                      n_process,

                                  ], [
                                      asr_info,
                                      open_asr_button,
                                      close_asr_button
                                  ])
            close_asr_button.click(close_asr, [], [asr_info,open_asr_button,close_asr_button])
            asr_model.change(change_lang_choices, [asr_model], [asr_lang])
            asr_model.change(change_size_choices, [asr_model], [asr_size])
            # open_slicer_button.click(open_slice, [slice_inp_path,slice_opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_process], [slicer_info,open_slicer_button,close_slicer_button])
            # close_slicer_button.click(close_slice, [], [slicer_info,open_slicer_button,close_slicer_button])
            # open_denoise_button.click(open_denoise, [denoise_input_dir,denoise_output_dir], [denoise_info,open_denoise_button,close_denoise_button])
            # close_denoise_button.click(close_denoise, [], [denoise_info,open_denoise_button,close_denoise_button])

        with gr.TabItem("1.5-训练语音文本校对"):
            path_list = gr.Textbox(
                label=i18n(".list标注文件的路径"),
                value=os.environ['download_path'] + "/output/asr_opt/denoise_opt.list",
                interactive=True,
            )
            # reBtn = gr.Button(
            #     "刷新",
            # )
            # path_list.change(change_label, [path_list])
            with gr.Blocks() as demo:
                with gr.Row():
                    btn_change_index = gr.Button("Change Index / Refresh")
                    btn_submit_change = gr.Button("Submit Text")
                    btn_merge_audio = gr.Button("Merge Audio")
                    btn_delete_audio = gr.Button("Delete Audio")
                    btn_previous_index = gr.Button("Previous Index")
                    btn_next_index = gr.Button("Next Index")

                with gr.Row():
                    index_slider = gr.Slider(
                        minimum=0, maximum=g_max_json_index, value=g_index, step=1, label="Index", scale=3
                    )
                    splitpoint_slider = gr.Slider(
                        minimum=0, maximum=120.0, value=0, step=0.1, label="Audio Split Point(s)", scale=3
                    )
                    btn_audio_split = gr.Button("Split Audio", scale=1)
                    btn_save_json = gr.Button("Save File", visible=True, scale=1)
                    btn_invert_selection = gr.Button("Invert Selection", scale=1)

                with gr.Row():
                    with gr.Column():
                        for _ in range(0, g_batch):
                            with gr.Row():
                                text = gr.Textbox(
                                    label="Text",
                                    visible=True,
                                    scale=5
                                )
                                audio_output = gr.Audio(
                                    label="Output Audio",
                                    visible=True,
                                    scale=5
                                )
                                audio_check = gr.Checkbox(
                                    label="Yes",
                                    show_label=True,
                                    info="Choose Audio",
                                    scale=1
                                )
                                g_text_list.append(text)
                                g_audio_list.append(audio_output)
                                g_checkbox_list.append(audio_check)

                with gr.Row():
                    batchsize_slider = gr.Slider(
                        minimum=1, maximum=g_batch, value=g_batch, step=1, label="Batch Size", scale=3,
                        interactive=False
                    )
                    interval_slider = gr.Slider(
                        minimum=0, maximum=2, value=0, step=0.01, label="Interval", scale=3
                    )
                    btn_theme_dark = gr.Button("Light Theme", link="?__theme=light", scale=1)
                    btn_theme_light = gr.Button("Dark Theme", link="?__theme=dark", scale=1)

                btn_change_index.click(
                    change_label,
                    inputs=[
                        path_list,
                        index_slider,
                        batchsize_slider,
                    ],
                    outputs=[
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ],
                )

                btn_submit_change.click(
                    b_submit_change,
                    inputs=[
                        *g_text_list,
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ],
                )

                btn_previous_index.click(
                    b_previous_index,
                    inputs=[
                        index_slider,
                        batchsize_slider,
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ],
                )

                btn_next_index.click(
                    b_next_index,
                    inputs=[
                        index_slider,
                        batchsize_slider,
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ],
                )

                btn_delete_audio.click(
                    b_delete_audio,
                    inputs=[
                        *g_checkbox_list
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ]
                )

                btn_merge_audio.click(
                    b_merge_audio,
                    inputs=[
                        interval_slider,
                        *g_checkbox_list
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ]
                )

                btn_audio_split.click(
                    b_audio_split,
                    inputs=[
                        splitpoint_slider,
                        *g_checkbox_list
                    ],
                    outputs=[
                        index_slider,
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ]
                )

                btn_invert_selection.click(
                    b_invert_selection,
                    inputs=[
                        *g_checkbox_list
                    ],
                    outputs=[
                        *g_checkbox_list
                    ]
                )

                btn_save_json.click(
                    b_save_file
                )

                demo.load(
                    b_change_index,
                    inputs=[
                        index_slider,
                        batchsize_slider,
                    ],
                    outputs=[
                        *g_text_list,
                        *g_audio_list,
                        *g_checkbox_list
                    ],
                )
        with gr.TabItem(i18n("2-模型微调")):
            gr.Markdown(value='## ' + i18n("SoVITS训练"))
            gr.Markdown(value=i18n('用于分享的模型文件输出在SoVITS_weights下。'))
            with gr.Row():
                batch_size = gr.Slider(minimum=1, maximum=40, step=1, label=i18n("每张显卡的batch_size"),
                                       value=default_batch_size, interactive=True)
                total_epoch = gr.Slider(minimum=1, maximum=25, step=1, label=i18n("总训练轮数total_epoch，不建议太高"),
                                        value=8, interactive=True)
                text_low_lr_rate = gr.Slider(minimum=0.2, maximum=0.6, step=0.05, label=i18n("文本模块学习率权重"),
                                             value=0.4, interactive=True)
                save_every_epoch = gr.Slider(minimum=1, maximum=25, step=1, label=i18n("保存频率save_every_epoch"),
                                             value=4, interactive=True)
                if_save_latest = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True,
                                             interactive=True, show_label=True)
                if_save_every_weights = gr.Checkbox(label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                                    value=True, interactive=True, show_label=True)
                gpu_numbers1Ba = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus),
                                            interactive=True)
            with gr.Row():
                button1Ba_open = gr.Button(i18n("开启SoVITS训练"), variant="primary", visible=True)
                button1Ba_close = gr.Button(i18n("终止SoVITS训练"), variant="primary", visible=False)
                info1Ba = gr.Textbox(label=i18n("SoVITS训练进程输出信息"))
            gr.Markdown(value='## ' + i18n("GPT训练"))
            gr.Markdown(value=i18n('用于分享的模型文件输出在GPT_weights下。'))
            with gr.Row():
                batch_size1Bb = gr.Slider(minimum=1, maximum=40, step=1, label=i18n("每张显卡的batch_size"),
                                          value=default_batch_size, interactive=True)
                total_epoch1Bb = gr.Slider(minimum=2, maximum=50, step=1, label=i18n("总训练轮数total_epoch"), value=15,
                                           interactive=True)
                if_dpo = gr.Checkbox(label=i18n("是否开启dpo训练选项(实验性)"), value=False, interactive=True,
                                     show_label=True)
                if_save_latest1Bb = gr.Checkbox(label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"), value=True,
                                                interactive=True, show_label=True)
                if_save_every_weights1Bb = gr.Checkbox(
                    label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"), value=True, interactive=True,
                    show_label=True)
                save_every_epoch1Bb = gr.Slider(minimum=1, maximum=50, step=1, label=i18n("保存频率save_every_epoch"),
                                                value=5, interactive=True)
                gpu_numbers1Bb = gr.Textbox(label=i18n("GPU卡号以-分割，每个卡号一个进程"), value="%s" % (gpus),
                                            interactive=True)
            with gr.Row():
                button1Bb_open = gr.Button(i18n("开启GPT训练"), variant="primary", visible=True)
                button1Bb_close = gr.Button(i18n("终止GPT训练"), variant="primary", visible=False)
                info1Bb = gr.Textbox(label=i18n("GPT训练进程输出信息"))
            gr.Markdown(value='## ' + i18n("参数设置"))
            with gr.Row():
                bert_pretrained_dir = gr.Textbox(label=i18n("预训练的中文BERT模型路径"),
                                                 value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                                                 interactive=False)
                cnhubert_base_dir = gr.Textbox(label=i18n("预训练的SSL模型路径"),
                                               value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                               interactive=False)
                inp_text = gr.Textbox(label=i18n("*文本标注文件"),
                                      value=os.environ['download_path'] + "/output/asr_opt/denoise_opt.list",
                                      interactive=True)
                inp_wav_dir = gr.Textbox(
                    label=i18n("*训练集音频文件目录"),
                    # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                    interactive=True,
                    placeholder=i18n(
                        "填切割后音频所在目录！读取的音频文件完整路径=该目录-拼接-list文件里波形对应的文件名（不是全路径）。如果留空则使用.list文件里的绝对全路径。")
                )
            with gr.Row():
                exp_name = gr.Textbox(label=i18n("*实验/模型名"), value="xxx", interactive=True)
                gpu_info = gr.Textbox(label=i18n("显卡信息"), value=gpu_info, visible=True, interactive=False)
                pretrained_s2G = gr.Textbox(label=i18n("预训练的SoVITS-G模型路径"),
                                            value="GPT_SoVITS/pretrained_models/s2G488k.pth", interactive=True)
                pretrained_s2D = gr.Textbox(label=i18n("预训练的SoVITS-D模型路径"),
                                            value="GPT_SoVITS/pretrained_models/s2D488k.pth", interactive=True)
                pretrained_s1 = gr.Textbox(label=i18n("预训练的GPT模型路径"),
                                           value="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                                           interactive=True)

            button1Ba_open.click(open1Ba,
                                 [
                                     batch_size,
                                     total_epoch,
                                     exp_name,
                                     text_low_lr_rate,
                                     if_save_latest,
                                     if_save_every_weights,
                                     save_every_epoch,
                                     gpu_numbers1Ba,
                                     pretrained_s2G,
                                     pretrained_s2D,

                                     inp_text,
                                     inp_wav_dir,
                                     bert_pretrained_dir,
                                     cnhubert_base_dir,
                                 ], [info1Ba, button1Ba_open, button1Ba_close])
            button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])
            button1Bb_open.click(open1Bb,
                                 [
                                     batch_size1Bb,
                                     total_epoch1Bb,
                                     exp_name, if_dpo,
                                     if_save_latest1Bb,
                                     if_save_every_weights1Bb,
                                     save_every_epoch1Bb,
                                     gpu_numbers1Bb,
                                     pretrained_s1,

                                     pretrained_s2G,
                                     inp_text,
                                     inp_wav_dir,
                                     bert_pretrained_dir,
                                     cnhubert_base_dir,
                                 ], [info1Bb, button1Bb_open, button1Bb_close])
            button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])
        with gr.TabItem(i18n("3-GPT-SoVITS-变声")):gr.Markdown(value=i18n("施工中，请静候佳音"))
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        quiet=True,
    )
