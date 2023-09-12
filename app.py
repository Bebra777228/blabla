import os, sys
import datetime, subprocess
from mega import Mega
now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import shutil
import threading
import traceback
import warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib

import fairseq
import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from infer.modules.uvr5.modules import uvr
from infer.modules.vc.modules import VC
logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


load_dotenv()
config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("Мне жаль, что у вас нет рабочей видеокарты для поддержки обучения.")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg')):
            audio_files.append('./audios/'+filename)
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }, {"choices": sorted(audio_files), "__type__": "update"}

def clean():
    return {"value": "", "__type__": "update"}


def export_onnx():
    from infer.modules.onnx.export import export_onnx as eo

    eo()


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warn(
            "assets/pretrained%s/%sG%s.pth не существует, не будет использоваться предварительно обученная модель",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warn(
            "assets/pretrained%s/%sD%s.pth не существует, не будет использоваться предварительно обученная модель",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "По окончании обучения можно просмотреть журнал обучения на консоли или журнал train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Пожалуйста, сначала сделайте извлечение функций!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Пожалуйста, сначала выполните извлечение тона!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    ####### step1:处理数据
    yield get_info_str(i18n("шаг 1: Обработка данных"))
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    ####### step2a:提取音高
    yield get_info_str(i18n("шаг 2: Извлечение высоты тона"))
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    ####### step3a:训练模型
    yield get_info_str(i18n("шаг 3: Тренировать модель"))
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    )
    yield get_info_str(i18n("По окончании обучения можно просмотреть журнал обучения на консоли или журнал train.log"))

    ####### step3b:训练索引
    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str(i18n("Конец полного процесса!"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}

def find_model():
    if len(names) > 0:
        vc.get_vc(sorted(names)[0],None,None)
        return sorted(names)[0]
    else:
        try:
            gr.Info("Do not forget to choose a model.")
        except:
            pass
        return ''
    
def find_audios(index=False):     
    audio_files=[]
    if not os.path.exists('./audios'): os.mkdir("./audios")
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg')):
            audio_files.append("./audios/"+filename)
    if index:
        if len(audio_files) > 0: return sorted(audio_files)[0]
        else: return ""
    elif len(audio_files) > 0: return sorted(audio_files)
    else: return []

def get_index():
    if find_model() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
        
def get_indexes():
    indexes_list=[]
    for dirpath, dirnames, filenames in os.walk("./logs/"):
        for filename in filenames:
            if filename.endswith(".index"):
                indexes_list.append(os.path.join(dirpath,filename))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''
    
def save_wav(file):
    try:
        file_path=file.name
        shutil.move(file_path,'./audios')
        return './audios/'+os.path.basename(file_path)
    except AttributeError:
        try:
            new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
            new_path='./audios/'+new_name
            shutil.move(file,new_path)
            return new_path
        except TypeError:
            return None

def download_from_url(url, model):
    if url == '':
        return "URL нельзя оставлять пустым."
    if model =='':
        return "Необходимо дать имя своей модели. Например: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./assets/weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Успех."
    except:
        return "Произошла ошибка."

def upload_to_dataset(files, dir):
    if dir == '':
        dir = './dataset/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        path=file.name
        shutil.copy2(path,dir)
    try:
        gr.Info(i18n("Обработка данных"))
    except:
        pass
    return i18n("Обработка данных"), {"value":dir,"__type__":"update"}

def download_model_files(model):
    model_found = False
    index_found = False
    if os.path.exists(f'./assets/weights/{model}.pth'): model_found = True
    if os.path.exists(f'./logs/{model}'):
        for file in os.listdir(f'./logs/{model}'):
            if file.endswith('.index') and 'added' in file:
                log_file = file
                index_found = True
    if model_found and index_found:
        return [f'./assets/weights/{model}.pth', f'./logs/{model}/{log_file}'], "Done"
    elif model_found and not index_found:
        return f'./assets/weights/{model}.pth', "Не удалось найти файл Index."
    elif index_found and not model_found:
        return f'./logs/{model}/{log_file}', f'Убедитесь, что имя голоса указано правильно. Я не могу найти {model}.pth'
    else:
        return None, f'Не удалось найти {model}.pth или соответствующий индексный файл.'

with gr.Blocks(title="🔊",theme=gr.themes.Base(primary_hue="rose",neutral_hue="zinc")) as app:
    with gr.Row():
        gr.HTML("<img  src='file/a.png' alt='image'>")
    with gr.Tabs():
        with gr.TabItem(i18n("Вывод модели")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("Ваша модель"), choices=sorted(names), value=find_model())
                refresh_button = gr.Button(i18n("Обновить"), variant="primary")
                #clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                #clean_button.click(
                #    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                #)
                vc_transform0 = gr.Number(
                    label=i18n("Transpose (целое число, количество полутонов, повышение на октаву: 12, понизить на октаву: -12):"), value=0
                )
                but0 = gr.Button(i18n("Конвертировать"), variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Отправьте аудиозапись сюда и нажмите кнопку Перезагрузка.")
                    with gr.Row():
                        record_button=gr.Audio(source="microphone", label="Запись звука с микрофона.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label=i18n("Введите путь к обрабатываемому аудиофайлу (по умолчанию - пример правильного формата)"),
                            value=find_audios(True),
                            choices=find_audios()
                        )
                        record_button.change(fn=save_wav, inputs=[record_button], outputs=[input_audio0])
                        dropbox.upload(fn=save_wav, inputs=[dropbox], outputs=[input_audio0])
                with gr.Column():
                    with gr.Accordion(label=i18n("Автоопределение пути индекса и выбор из выпадающего списка:"), open=False):
                        file_index2 = gr.Dropdown(
                            label=i18n("Файл индекса (выберите ваш индекс из списка если это не произошло автоматически)"),
                            choices=get_indexes(),
                            interactive=True,
                            value=get_index()
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Коэффициент поискового тона (управляет силой акцента, слишком высокий коэффициент приводит к появлению артефактов):"),
                            value=0.66,
                            interactive=True,
                        )
                    vc_output2 = gr.Audio(label=i18n("Экспорт аудио (нажмите на три точки в правом нижнем углу, чтобы загрузить)"))
                    with gr.Accordion(label=i18n("Общие настройки"), open=False):
                        f0method0 = gr.Radio(
                            label=i18n(
                                "Выберите алгоритм выделения высоты тона ('pm': более быстрое выделение, но низкое качество речи; 'harvest': лучший бас, но очень медленный; 'crepe': лучшее качество, но требует больших затрат GPU; 'rmvpe': лучшее качество и малые требования к GPU)"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n("Если >=3: применить медианную фильтрацию к собранным результатам питча. Значение представляет собой радиус фильтрации и может уменьшить дыхание."),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Постпроцессинговая передискретизация до конечной частоты дискретизации, 0 - без передискретизации"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Настройте масштаб огибающей громкости. Чем ближе к 0, тем больше она имитирует громкость оригинального вокала. При относительно низком уровне громкости может помочь замаскировать шумы и сделать звучание более естественным. Ближе к 1 - громкость будет более стабильной:"),
                            value=0.21,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Защита безголосых согласных и звуков дыхания для предотвращения артефактов, таких как разрывы в электронной музыке. Для отключения установите значение 0,5. Уменьшите значение для усиления защиты, но это может снизить точность индексирования:"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    file_index1 = gr.Textbox(
                        label=i18n("Путь к файлу библиотеки поиска тона, если он пуст, используйте выпадающий результат выбора."),
                        value="",
                        interactive=True,
                        visible=False
                    )
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[sid0, file_index2, input_audio0],
                        api_name="infer_refresh",
                    )
                    # file_big_npy1 = gr.Textbox(
                    #     label=i18n("特征文件路径"),
                    #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                    #     interactive=True,
                    # )
            with gr.Row():
                f0_file = gr.File(label=i18n("Файл кривой F0, необязательный, один шаг в строке, заменяет F0 и высоту по умолчанию."), visible=False)
            with gr.Row():
                vc_output1 = gr.Textbox(label=i18n("Выходная информация"))
                but0.click(
                    vc.vc_single,  
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
            with gr.Row():
                with gr.Accordion(open=False, label=i18n("Пакетное преобразование. Укажите папку, содержащую аудиофайлы для конвертирования, или загрузите несколько аудиофайлов. Преобразованные аудиофайлы будут выведены в указанную папку (по умолчанию: 'opt').")):                
                    with gr.Row():
                        opt_input = gr.Textbox(label=i18n("Укажите выходную папку:"), value="opt")
                        vc_transform1 = gr.Number(
                            label=i18n("Transpose (целое число, количество полутонов, повышение на октаву: 12, понизить на октаву: -12):"), value=0
                        )
                        f0method1 = gr.Radio(
                            label=i18n(
                                "Выберите алгоритм выделения высоты тона ('pm': более быстрое выделение, но низкое качество речи; 'harvest': лучший бас, но очень медленный; 'crepe': лучшее качество, но требует больших затрат GPU; 'rmvpe': лучшее качество и малые требования к GPU)"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="pm",
                            interactive=True,
                        )
                    with gr.Row():
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n("Если >=3: применить медианную фильтрацию к собранным результатам питча. Значение представляет собой радиус фильтрации и может уменьшить дыхание."),
                            value=3,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Row():
                        file_index3 = gr.Textbox(
                            label=i18n("Путь к файлу библиотеки поиска тона, если он пуст, используйте выпадающий результат выбора."),
                            value="",
                            interactive=True,
                            visible=False
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("Автоопределение пути индекса и выбор из выпадающего списка:"),
                            choices=sorted(index_paths),
                            interactive=True,
                            visible=False
                        )
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Процентное соотношение поисковых функций"),
                            value=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Row():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Постпроцессинговая передискретизация до конечной частоты дискретизации, 0 - без передискретизации"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Настройте масштаб огибающей громкости. Чем ближе к 0, тем больше она имитирует громкость оригинального вокала. При относительно низком уровне громкости может помочь замаскировать шумы и сделать звучание более естественным. Ближе к 1 - громкость будет более стабильной:"),
                            value=0.21,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Защита безголосых согласных и звуков дыхания для предотвращения артефактов, таких как разрывы в электронной музыке. Для отключения установите значение 0,5. Уменьшите значение для усиления защиты, но это может снизить точность индексирования:"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Row():
                        dir_input = gr.Textbox(
                            label=i18n("Введите путь к обрабатываемой папке с аудиофайлами (скопируйте его из адресной строки файлового менеджера):"),
                            value="./audios",
                        )
                        inputs = gr.File(
                            file_count="multiple", label=i18n("Можно также импортировать несколько аудиофайлов. Если путь к папке существует, то этот ввод игнорируется.")
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("Формат экспортного файла"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("Конвертировать"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("Выходная информация"))
                        but1.click(
                            vc.vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index1,
                                file_index2,
                                # file_big_npy2,
                                index_rate1,
                                filter_radius1,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                            ],
                            [vc_output3],
                            api_name="infer_convert_batch",
                        )
            sid0.change(
                fn=vc.get_vc,
                inputs=[sid0, protect0, protect1],
                outputs=[spk_item, protect0, protect1, file_index2, file_index4],
            )
        with gr.TabItem("Загрузить модель"):
            with gr.Row():
                url=gr.Textbox(label="Введите URL-адрес модели:")
            with gr.Row():
                model = gr.Textbox(label="Название модели:")
                download_button=gr.Button("Загрузить")
            with gr.Row():
                status_bar=gr.Textbox(label="")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
                """
                ❤️ If you use this and like it, help me keep it.❤️ 
                https://paypal.me/lesantillan
                """
                )
        with gr.TabItem(i18n("Тренировка")):
            with gr.Row():
                with gr.Column():
                    exp_dir1 = gr.Textbox(label=i18n("Введите имя модели:"), value="My-Voice")
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label=i18n("Количество процессов ЦП, используемых для извлечения питча и обработки данных"),
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    sr2 = gr.Radio(
                        label=i18n("Заданная частота дискретизации"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                        visible=False
                    )
                    if_f0_3 = gr.Radio(
                        label=i18n("Поставляется ли модель в комплекте с питч-гидом (пение - обязательно, голос - нет)"),
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    version19 = gr.Radio(
                        label=i18n("版本"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Укажите путь к учебной папке:"), value='./dataset/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    )
                    easy_uploader = gr.Files(label=i18n("Можно также импортировать несколько аудиофайлов. Если путь к папке существует, то этот ввод игнорируется."),file_types=['audio'])
                    but1 = gr.Button(i18n("шаг 1: Обработка данных"), variant="primary")
                    info1 = gr.Textbox(label=i18n("Выходная информация"), value="")
                    easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader, trainset_dir4], outputs=[info1, trainset_dir4])
                    gpus6 = gr.Textbox(
                        label=i18n("Введите индекс(ы) GPU, разделенные символом '-', например, 0-1-2 для использования GPU 0, 1 и 2:"),
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpu_info9 = gr.Textbox(
                        label=i18n("Информация о видеокарте"), value=gpu_info, visible=F0GPUVisible
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                        visible=False
                    )
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    ) 
                with gr.Column():
                    f0method8 = gr.Radio(
                        label=i18n(
                            "Выберите алгоритм выделения высоты тона ('pm': более быстрое выделение, но низкое качество речи; 'harvest': лучший бас, но очень медленный; 'dio': лучшее качество, но требует больших затрат GPU, 'rmvpe': лучшее качество но немного сьедает GPU/CPU)"
                        ),
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label=i18n(
                            "Введите индекс(ы) GPU, разделенные символом '-', например, 0-0-1, чтобы использовать 2 процесса на GPU0 и 1 процесс на GPU1"
                        ),
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    but2 = gr.Button(i18n("шаг 2: Извлечение питча"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Выходная информация"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
                with gr.Column():
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Общее количество эпох обучения (total_epoch):"),
                        value=150,
                        interactive=True,
                    )
                    gpus16 = gr.Textbox(
                            label=i18n("Введите индекс(ы) GPU, разделенные символом '-', например, 0-1-2 для использования GPU 0, 1 и 2:"),
                            value="0",
                            interactive=True,
                            visible=True
                        )
                    but3 = gr.Button(i18n("шаг 3: Тренировка модели"), variant="primary")
                    but4 = gr.Button(i18n("шаг 4: Тренировка индекса"), variant="primary")
                    info3 = gr.Textbox(label=i18n("Выходная информация"), value="", max_lines=10)
                    with gr.Accordion(label=i18n("Общие настройки"), open=False):
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("Частота сохранения (save_every_epoch):"),
                            value=25,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("Размер партии на один графический процессор:"),
                            value=13,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label=i18n("Сохранять ли только последний файл ckpt для экономии места на жестком диске"),
                            choices=[i18n("Да"), i18n("Нет")],
                            value=i18n("Да"),
                            interactive=True,
                            visible=False
                        )
                        if_cache_gpu17 = gr.Radio(
                            label=i18n(
                                "Кэшировать все обучающие наборы в память GPU. Кэширование небольших наборов данных (менее 10 минут) может ускорить обучение, но кэширование больших наборов данных будет занимать много памяти GPU и может не дать значительного улучшения скорости:"
                            ),
                            choices=[i18n("Да"), i18n("Нет")],
                            value=i18n("Нет"),
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label=i18n("В каждой точке сохранения сохраняйте небольшую итоговую модель в папке 'weights':"),
                            choices=[i18n("Да"), i18n("Нет")],
                            value=i18n("Да"),
                            interactive=True,
                        )
                    with gr.Row():
                        download_model = gr.Button('шаг 5: Скачать модель')
                    with gr.Row():
                        model_files = gr.Files(label='Файл с моделью и индексом можно загрузить здесь:')
                        download_model.click(fn=download_model_files, inputs=[exp_dir1], outputs=[model_files, info3])
                    with gr.Row():
                        pretrained_G14 = gr.Textbox(
                            label=i18n("加载预训练底模G路径"),
                            value="assets/pretrained_v2/f0G40k.pth",
                            interactive=True,
                            visible=False
                        )
                        pretrained_D15 = gr.Textbox(
                            label=i18n("加载预训练底模D路径"),
                            value="assets/pretrained_v2/f0D40k.pth",
                            interactive=True,
                            visible=False
                        )
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19],
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                    with gr.Row():
                        but5 = gr.Button(i18n("一键训练"), variant="primary", visible=False)
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                            api_name="train_start",
                        )
                        but4.click(train_index, [exp_dir1, version19], info3)
                        but5.click(
                            train1key,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                trainset_dir4,
                                spk_id5,
                                np7,
                                f0method8,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                                gpus_rmvpe,
                            ],
                            info3,
                            api_name="train_start_all",
                        )

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
