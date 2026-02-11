
import torch
def demo():
    file = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/asset/zero_shot_prompt.wav"
    pass

    from modelscope import snapshot_download
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/CosyVoice-ttsfrd')


def demo1():
    import torch
    all_pt_file = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/audio_decode_cosyvoice_debug.pt"
    data = torch.load(all_pt_file)
    # print(data)
    for key in data.keys(): 
        print(key)
        print(data[key])
        print(data[key].dtype)
    pass

def demo2():
    file = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4/spk2info.pt"
    data = torch.load(file)
    keys = list(data.keys())
    print(keys)
    val = data['001']
    val_keys = list(val.keys())
    print(val)
    pass

def demo3():
    import sys

    sys.path.insert(
        0, "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice"
    )
    sys.path.insert(
        0,
        "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS",
    )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import AutoModel
    # 保存speech 信息到spk2info.pt
    file = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4/spk2info.pt"
    cur_data = torch.load(file)
    print("原有speaker数量:", len(cur_data))


    token2wav_path = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4"
    token2wav_model = AutoModel(
        model_dir=token2wav_path, load_trt=False, fp16=False
    )

    prompt_text = "Whether you walking, working or doing gym, just put on your headphones."
    prompt_speech_path = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Arif.wav_0002947200_0003104640_16k.wav"
    model_inputs_dict = token2wav_model.frontend.frontend_zero_shot(
        "tts_text", prompt_text, prompt_speech_path, 24000, ""
    )

    spk_name = "Arif"
    prompt_wav = prompt_speech_path
    cur_data[spk_name] = {
        "spk_id": spk_name,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
        "prompt_token": model_inputs_dict["flow_prompt_speech_token"],
        "prompt_feat": model_inputs_dict["prompt_speech_feat"],
        "embedding": model_inputs_dict["flow_embedding"],
    }
    print("更新后speaker数量:", len(cur_data))
    torch.save(cur_data, file)



if __name__ == "__main__":
    demo3()


"""
model_repo=./model_repo_cosyvoice2
tritonserver --model-repository ./model_repo_cosyvoice3_v2
tritonserver --model-repository ./model_repo_cosyvoice2

# install FlashCosyVoice for token2wav batching
# git clone https://github.com/yuekaizhang/FlashCosyVoice.git /workspace/FlashCosyVoice -b trt
# cd /workspace/FlashCosyVoice
# pip install -e .
# cd -
# wget https://huggingface.co/yuekai/cosyvoice2_flow_onnx/resolve/main/flow.decoder.estimator.fp32.dynamic_batch.onnx -O $model_scope_model_local_dir/flow.decoder.estimator.fp32.dynamic_batch.onnx

bash run.sh 6 6

# You can also switch to huggingface backend by setting backend=hf

import sys
sys.path.append("/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice")
sys.path.append("/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS")

apt install ffmpeg -y

/gpfs01/nfs_share/data20250106/panxb/pretrained_models/Fun-CosyVoice3-0.5B

huggingface-cli download --resume-download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./Fun-CosyVoice3-0.5B-2512

pip install inflect
pip install loguru
pip install x-transformers
pip install black==25.1.0
pip install 
pip install 
pip install 
pip install 
pip install 
pip install 
pip install 
pip install 
pip install 

ps -ef | grep tritonserver| grep -v grep | cut -c 9-16 | xargs kill -9
ps -ef | grep model.py| grep -v grep | cut -c 9-16 | xargs kill -9


nohup  tritonserver --model-repository ./model_repo_cosyvoice3_v2 > triton_cosyvoice3_v2.log 2>&1 &
tail -f triton_cosyvoice3_v2.log
ps aux|grep tritonserver

export PYTHONPATH=/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice:$PYTHONPATH


nohup  python3  /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/example.py > 0204_example.log 2>&1 &


cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

tar -czf output_audio_v3.tar.gz output_audio_v3/

==================================================
cosy3新模型加速步骤
pip install inflect
pip install loguru
pip install x-transformers
pip install black==25.1.0

cd /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3

export PYTHONPATH=/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice:$PYTHONPATH

1. 
python3 convert_cosyvoice3_to_hf.py --model-dir /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4 --output-dir /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/transformers_cosyvoice3_llm_zh_en_malay_spanish_arabic_singlish_data_v4

--model-dir: cosyvoice3原始模型目录
--output-dir: 转换后huggingface模型目录
2. 
修改/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/model_repo_cosyvoice3_v2/cosyvoice3/config.pbtxt下的路径

3.
转化成trt形式，修改run_panxb_v3.sh中的huggingface_model_local_dir,trt_weights_dir,trt_engines_dir变量为新的路径，然后运行bash run_panxb_v3.sh 1 1

4. 
tritonserver --model-repository ./model_repo_cosyvoice3_v2


tar -czf /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v4_compare.tar.gz /gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/train_multi_language/cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v4_compare


"""