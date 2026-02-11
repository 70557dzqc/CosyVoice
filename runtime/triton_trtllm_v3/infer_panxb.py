import os
import soundfile as sf
import numpy as np
import argparse
import requests
import time


def prepare_request(
    waveform,
    reference_text,
    target_text,
    sample_rate=16000,
    padding_duration: int = None,
    audio_save_dir: str = "./",
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    if padding_duration:
        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration
                * sample_rate
                * ((int(len(waveform) / sample_rate) // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform
    else:
        samples = waveform

    samples = samples.reshape(1, -1).astype(np.float32)
    # reference_text = f"You are a helpful assistant.<|endofprompt|>{reference_text}"
    print(f"reference_text: {reference_text}")

    data = {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist(),
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text],
            },
        ]
    }

    return data


txt_ = "/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/train_multi_language/2_test_multi.txt"
# instruct_ = False
instruct_ = True
## with open("test_multi_20251223.txt", 'r', encoding='utf-8') as f:
with open(txt_, "r", encoding="utf-8") as f:
    lines = [i.strip() for i in f.readlines()]

cnt = 0
for i, line in enumerate(lines):
    if "/malay/" not in line:
        continue
    if cnt != 0:
        continue
        # pass
    cnt += 1
    try:
        parts = line.split("|", 4)
        wav_path, gen_wav_path, content, gen_text = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )
    except:
        parts = line.split("|", 5)
        wav_path, gen_wav_path, content, gen_text, qwen_txt = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
        )
    wav_path = f"/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/{wav_path}"
    print("wav_path:", wav_path)
    # 统计采样率
    audio_info = sf.info(wav_path)
    print("采样率:", audio_info.samplerate)
    if audio_info.samplerate != 16000:
        wav_path = f"/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/{os.path.basename(wav_path)[:-4]}_16k.wav"
        assert os.path.exists(wav_path)
        print("转换后16k采样率音频路径:", wav_path)
        audio_info = sf.info(wav_path)
        print("转换后采样率:", audio_info.samplerate)
    print(f"content: {content}")
    print(f"gen_text: {gen_text}")
    
    # # 转到16k采样率
    # if audio_info.samplerate != 16000:
    #     import torchaudio

    #     waveform, sample_rate = torchaudio.load(wav_path)
    #     resampler = torchaudio.transforms.Resample(
    #         orig_freq=sample_rate, new_freq=16000
    #     )
    #     waveform = resampler(waveform)

    if not os.path.exists(wav_path):
        continue

    # if instruct_:

    #     prompt = "You are a helpful assistant.<|endofprompt|>"
    #     content = f"{prompt}{content}"

    prefix = "_jiasu_0211_try2"

    gen_wav_path = gen_wav_path.replace(
        "cosy3_1_test_multi_20251224",
        f"cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v4{prefix}",
    )
    out_dir = os.path.dirname(gen_wav_path)  # 相对路径
    # print("gen_wav_path:", gen_wav_path)
    # print("out_dir:", out_dir)
    out_dir = f"/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/{out_dir}"
    gen_wav_path = f"/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/{gen_wav_path[:-4]}{prefix}.wav"


    # print("out_dir full path:", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(gen_wav_path):
        # continue
        pass

    url = "http://localhost:8000/v2/models/cosyvoice3_spk/infer"

    waveform, sr = sf.read(wav_path)
    assert sr == 16000, "sample rate hardcoded in server"
    samples = np.array(waveform, dtype=np.float32)
    data = prepare_request(samples, content, gen_text)

    st_time = time.time()
    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": "0"},
    )
    end_time = time.time()
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)
    sample_rate = 24000
    # save_file = args.output_audio
    save_file = gen_wav_path
    print(f"Saving generated audio to {save_file}")
    sf.write(save_file, audio, sample_rate, "PCM_16")
    duration = len(audio) / sample_rate
    rtf = (end_time - st_time) / duration
    print(
        f"Total time: {end_time - st_time:.3f}s, Audio duration: {duration:.3f}s, RTF: {rtf:.3f}"
    )
