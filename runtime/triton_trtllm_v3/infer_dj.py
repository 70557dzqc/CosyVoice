import sys
sys.path.append('/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2,CosyVoice3,AutoModel
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import os


# # Convert the response to a single audio stream
#    tts_audio = b''.join(response)
#
#    # Create an in-memory buffer to store the audio data
#    buf = io.BytesIO()
#
#    # Create WAV header
#    sample_rate = 22050  # 假设采样率为22050Hz
#    num_channels = 1  # 单声道
#    bits_per_sample = 16  # 16位采样
#    byte_rate = sample_rate * num_channels * bits_per_sample // 8
#    block_align = num_channels * bits_per_sample // 8
#
#    # Write WAV header
#    buf.write(b'RIFF')
#    buf.write((36 + len(tts_audio)).to_bytes(4, 'little'))  # 文件大小
#    buf.write(b'WAVE')
#    buf.write(b'fmt ')
#    buf.write((16).to_bytes(4, 'little'))  # 子块大小
#    buf.write((1).to_bytes(2, 'little'))  # PCM格式
#    buf.write((num_channels).to_bytes(2, 'little'))  # 通道数
#    buf.write((sample_rate).to_bytes(4, 'little'))  # 采样率
#    buf.write((byte_rate).to_bytes(4, 'little'))  # 字节率
#    buf.write((block_align).to_bytes(2, 'little'))  # 块对齐
#    buf.write((bits_per_sample).to_bytes(2, 'little'))  # 每个样本的位数
#    buf.write(b'data')
#    buf.write(len(tts_audio).to_bytes(4, 'little'))  # 数据大小
#
#    # Write audio data
#    buf.write(tts_audio)
#    buf.seek(0)
#
#    # Return the audio stream as a StreamingResponse
#    return StreamingResponse(buf, media_type="audio/wav")


# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B-spanish', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B-multi', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B-multi-update-1', load_jit=False, load_trt=False, load_vllm=False, fp16=False)


# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
# prompt_speech_16k = load_wav('train_multi_language/out-wavs/arabic_day1121_prompt02.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot("الفئة الثانية من المقاعد بدركسون إم بعدادها الديجيتال والفل.","بطاقة جمركية لم تستخدم في السعودية بلونها الأسود المثالي.", prompt_speech_16k, stream=False,text_frontend=False)):
#     # print(j)
#     torchaudio.save(f"train_multi_language/out-wavs/ara_ara-2.wav", j['tts_speech'], cosyvoice.sample_rate)

## 批量推理多语种的文本
# with open("test_multi.txt", 'r', encoding='utf-8') as f:
#     lines = [i.strip() for i in f.readlines()]
# for line in lines:
#     print(line)
    
#     parts = line.split('|', 4)
#     if 'ara_ara' in line:
#         # continue
#         print(parts)
#     wav_path, gen_wav_path, content, gen_text = parts[0], parts[1], parts[2], parts[3]
#     ##泰语要变一下目录
#     gen_wav_path = gen_wav_path.replace('train_multi_language','train_thai_language')
#     out_dir = os.path.dirname(gen_wav_path)
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     if os.path.exists(gen_wav_path):
#         continue
#     prompt_speech_16k = load_wav(wav_path, 16000)
#     for i, j in enumerate(cosyvoice.inference_zero_shot(gen_text, content, prompt_speech_16k, stream=False,text_frontend=False)):

#         torchaudio.save(gen_wav_path, j['tts_speech'], cosyvoice.sample_rate)

# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)




# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-multi')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-multi-instruct-20251224')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-multi-v2-20251225')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-singlish-v2-20251225')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-4-language-v2-20251226')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-4-language-v3-20251230')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-4-language-v4-20260104-instruct')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B-4-language-v4-20260105-instruct-official')
# cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B-thai')
# cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B-spanish')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v2_test_s3_test_cam')
# cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v3')
cosyvoice = AutoModel(model_dir='/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4')
# print(cosyvoice)


# zero_shot usage
# for i, j in enumerate(cosyvoice.inference_zero_shot('老板们都喜欢什么车呀？跟主播说说呗，主播可以一起帮忙找车，一定让你心满意足！', 'You are a helpful assistant.<|endofprompt|>那问题来了啊，很多朋友们都问咱们这台车有没有置换补贴，答案是咱们这车真的有，而且很大。',
#                                                     '/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/GLM-TTS/examples/prompt/pengfei_DJI_06_20250526_113506_00010.wav', stream=False)):
#     print(j['tts_speech'].shape)
#     torchaudio.save('cosyvoice3.wav', j['tts_speech'], cosyvoice.sample_rate)


# for i, j in enumerate(cosyvoice.inference_zero_shot('老板们都喜欢什么车呀？跟主播说说呗，主播可以一起帮忙找车，一定让你心满意足！', '那问题来了啊，很多朋友们都问咱们这台车有没有置换补贴，答案是咱们这车真的有，而且很大。',
#                                                     '/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/GLM-TTS/examples/prompt/pengfei_DJI_06_20250526_113506_00010.wav', stream=False)):
#     print(j['tts_speech'].shape)
#     torchaudio.save('cosyvoice3.wav', j['tts_speech'], cosyvoice.sample_rate)



## 批量推理多语种的文本  1_test_multi_inside.txt
# txt_ = "train_multi_language/1_test_multi_inside.txt"
# txt_ = "train_multi_language/1_test_multi.txt"
txt_ = "/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/train_multi_language/2_test_multi.txt"
# instruct_ = False
instruct_ = True
## with open("test_multi_20251223.txt", 'r', encoding='utf-8') as f:
with open(txt_, 'r', encoding='utf-8') as f:
    lines = [i.strip() for i in f.readlines()]

for line in lines:
    # print(line)
    if "/malay/" not in line:
        continue
    try:
        parts = line.split('|', 4)
        wav_path, gen_wav_path, content, gen_text = parts[0], parts[1], parts[2], parts[3]
    except:
        parts = line.split('|', 5)
        wav_path, gen_wav_path, content, gen_text, qwen_txt = parts[0], parts[1], parts[2], parts[3], parts[4]
    
    wav_path = f"/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/{wav_path}"
    if not os.path.exists(wav_path):
        continue
    # if 'zh_zh-' not in gen_wav_path:
    #     continue
    if instruct_:
        # if 'thai' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用泰语表达。<|endofprompt|>'
        # elif 'spanish' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用西班牙语表达。<|endofprompt|>'
        # elif 'malay' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用马来西亚语表达。<|endofprompt|>'
        # elif 'pu_pu' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用葡萄牙语表达。<|endofprompt|>'
        # elif 'ara_ara' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用沙特阿拉伯语表达。<|endofprompt|>'
        # elif 'singlish' in gen_wav_path:
        #     prompt= 'You are a helpful assistant. 请用新加坡语表达。<|endofprompt|>'
        # else:
        #     print(f'erro in processing {gen_wav_path}')
        prompt = 'You are a helpful assistant.<|endofprompt|>'
        content = f"{prompt}{content}"
    ##泰语要变一下目录
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_instruct_inside_20251225')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_instruct_20251224-1')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_instruct_inside_20251225-speed0.9')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_v2_20251225')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_v2_inside_20251225')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_multi_v2_inside_20251225-speed0.9')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_thai_20251225-speed0.92')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_spanish_20251225-speed0.92')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_official_20251225')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_singlish_v2_20251225')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_4_language-v3-20251230')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_4_language-v4-20250104-instruct')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_4_language-v4-20250105-instruct-official')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3-out-wavs-20251217')
    # gen_wav_path = gen_wav_path.replace('train_multi_language','train_thai_language')




    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_1_test_zh_en_malay_spanish_arabic_singlish_data_v2_test_s3_test_cam')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v2_test_s3_test_cam')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v2_test_s3_test_cam-stream')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v3')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v3-stream')
    gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v4_wo_jiasu')
    # gen_wav_path = gen_wav_path.replace('cosy3_1_test_multi_20251224', 'cosy3_2_test_zh_en_malay_spanish_arabic_singlish_data_v4-stream')
    out_dir = os.path.dirname(gen_wav_path)
    out_dir = f"/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/{out_dir}"
    gen_wav_path = f"/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/output_audio_v3/{gen_wav_path[:-4]}_wo_jiasu.wav"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # if os.path.exists(gen_wav_path):
    #     continue

    all_audio_chunks = []
    for i, j in enumerate(cosyvoice.inference_zero_shot(gen_text, content,
                                                        wav_path, stream=False, text_frontend=False, speed=1.0)):
        # print(j['tts_speech'].shape)
        all_audio_chunks.append(j['tts_speech'])
        # torchaudio.save(gen_wav_path, j['tts_speech'], cosyvoice.sample_rate)
    if all_audio_chunks:
        complete_audio = torch.cat(all_audio_chunks, dim=-1)
        # 保存拼接后的完整音频文件
        torchaudio.save(gen_wav_path, complete_audio, cosyvoice.sample_rate)
