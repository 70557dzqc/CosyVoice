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
from argparse import ArgumentParser
import numpy as np
import s3tokenizer
import soundfile as sf
from loguru import logger

ORIGINAL_VOCAB_SIZE = 151924


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4",
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="希望你以后能够做的比我还好呦。",
        help="The prompt text",
    )
    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/asset/zero_shot_prompt_16000.wav",
        help="The path to the prompt speech",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        help="The input text",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        # default='/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/transformers_cosyvoice3_llm_v2',
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/transformers_cosyvoice3_llm_zh_en_malay_spanish_arabic_singlish_data_v3",
        help="The path to the model",
    )
    parser.add_argument(
        "--speech_tokenizer_model_path",
        type=str,
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx",
        help="path to speech_tokenizer_v3.onnx",
    )

    args = parser.parse_args()
    return args


def audio_decode_cosyvoice(
    audio_tokens, tts_text, prompt_text, prompt_speech_path, codec_decoder
):

    # 函数的参数要与当前版本的cosy代码对上
    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        tts_text, prompt_text, prompt_speech_path, 24000, ""
    )
    logger.info(f"audio_tokens: {audio_tokens}")
    logger.info(f"prompt_speech_tokens: {model_inputs_dict["flow_prompt_speech_token"]}")
    logger.info(f"prompt_speech_feat: {model_inputs_dict['prompt_speech_feat']}")
    logger.info(f"prompt_spk_embedding: {model_inputs_dict['flow_embedding']}")
    # 数据类型
    logger.info(f"audio_tokens.dtype: {audio_tokens.dtype}")
    logger.info(f"prompt_speech_tokens.dtype: {model_inputs_dict['flow_prompt_speech_token'].dtype}")
    logger.info(f"prompt_speech_feat.dtype: {model_inputs_dict['prompt_speech_feat'].dtype}")
    logger.info(f"prompt_spk_embedding.dtype: {model_inputs_dict['flow_embedding'].dtype}")

    prompt_speech_tokens = model_inputs_dict["flow_prompt_speech_token"]
    prompt_speech_feat = model_inputs_dict["prompt_speech_feat"]
    prompt_spk_embedding = model_inputs_dict["flow_embedding"]
    # 保存到本地
    all_pt_file = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/audio_decode_cosyvoice_debug.pt"
    torch.save({
        "audio_tokens": audio_tokens,
        "prompt_speech_tokens": prompt_speech_tokens,
        "prompt_speech_feat": prompt_speech_feat,
        "prompt_spk_embedding": prompt_spk_embedding
    }, all_pt_file)
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(
            codec_decoder.model.device
        ),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to(
            codec_decoder.model.device
        ),
        prompt_token_len=model_inputs_dict["flow_prompt_speech_token_len"].to(
            codec_decoder.model.device
        ),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(
            codec_decoder.model.device
        ),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(
            codec_decoder.model.device
        ),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
        streaming=False,
    )
    # logger.info(f"tts_mel: {tts_mel}")
    # 这里v2与v3不一样了
    audio_hat, _ = codec_decoder.model.hift.inference(
        # speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
        speech_feat=tts_mel,
        finalize=True,
    )
    # logger.info(f"audio_hat: {audio_hat}")

    return audio_hat


def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda")

    args.model_path = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/transformers_cosyvoice3_llm_zh_en_malay_spanish_arabic_singlish_data_v4"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    args.token2wav_path = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B_zh_en_malay_spanish_arabic_singlish_data_v4"
    token2wav_model = AutoModel(
        model_dir=args.token2wav_path, load_trt=False, fp16=False
    )

    audio_tokenizer = s3tokenizer.load_model(args.speech_tokenizer_model_path).to(
        device
    )

    args.prompt_speech_path = "/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Arif.wav_0002947200_0003104640_16k.wav"
    args.prompt_text = (
        "Whether you walking, working or doing gym, just put on your headphones."
    )
    # audio_tokenizer
    waveform, sr = sf.read(args.prompt_speech_path)

    save_list = []
    text_list = [
        "Sebab semua polisi harga terbaik dan aktiviti menarik hari ni, khusus untuk follower je.Korang yang baru follow, kalau rasa konten saya okay, jangan lupa tekan like sokong sikit ya.Lepas tu, atas harga terbaik tu, ada lagi tambahan diskaun eksklusif untuk viewer live je.Ok, habis cakap semua ni, korang rasa macam mana?Cepat klik link dan daftar, senang je nanti boleh drive kereta baru.",
        "Sebab semua polisi harga terbaik dan aktiviti menarik hari ni, khusus untuk follower je.Korang yang baru follow, kalau rasa konten saya okay, jangan lupa tekan like sokong sikit ya.",
        "Lepas tu, atas harga terbaik tu, ada lagi tambahan diskaun eksklusif untuk viewer live je.Ok, habis cakap semua ni, korang rasa macam mana?Cepat klik link dan daftar, senang je nanti boleh drive kereta baru.",
    ]
    for i, input_text in enumerate(text_list):
        if i != 0:
            continue
        args.input_text = input_text
        for ix in range(1):
            mels = []
            wav_array = torch.from_numpy(np.array(waveform, dtype=np.float32)).to(
                device
            )
            wav_len = len(wav_array)
            wav = wav_array.squeeze(0)
            mels.append(s3tokenizer.log_mel_spectrogram(wav))

            mels, mels_lens = s3tokenizer.padding(mels)
            codes, codes_lens = audio_tokenizer.quantize(
                mels.to(device), mels_lens.to(device)
            )
            logger.info(f"codes: {codes}")
            codes = codes.clone()  # + ORIGINAL_VOCAB_SIZE
            prompt_speech_tokens = codes[0, : codes_lens[0].item()]
            prompt_speech_tokens = prompt_speech_tokens.cpu().numpy().tolist()
            # logger.info(f"prompt_speech_tokens: {prompt_speech_tokens}")
            prompt_speech_str = "".join([f"<|s_{t}|>" for t in prompt_speech_tokens])
            # logger.info(f"prompt_speech_str: {prompt_speech_str}")

            with torch.no_grad():
                # # Tokenize the text
                # chat = [
                #     {"role": "user", "content": f"{args.input_text}"},
                #     {"role": "assistant", "content": ""}
                # ]
                # if 'system' in tokenizer.chat_template:
                #     tokenizer.chat_template = TEMPLATE
                # input_ids = tokenizer.apply_chat_template(
                #     chat,
                #     tokenize=True,
                #     return_tensors='pt',
                #     continue_final_message=True
                # )
                # input_ids = input_ids.to(device)

                # https://github.com/Brakanier/FastCosyVoice/blob/main/fastcosyvoice/cosyvoice.py#L1046
                # <|s_6561|> = sos, <|s_6563|> = task_id
                prompt = f"<|s_6561|>You are a helpful assistant.<|endofprompt|>{args.prompt_text + args.input_text}<|s_6563|>{prompt_speech_str}"
                # logger.info(f"prompt: {prompt}")
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                # logger.info(f"input_ids: {input_ids}")
                # logger.info(f"input_ids.shape: {input_ids.shape}")

                # Generate the speech autoregressively
                outputs = model.generate(
                    input_ids,
                    max_length=2048,  # We trained our model with a max length of 2048
                    do_sample=True,  # True False
                    top_p=0.95,  #  Adjusts the diversity of generated content
                    temperature=0.8,  #  Controls randomness in output,
                    repetition_penalty=1.1,
                    top_k=3,  # 设置25容易多出一段重复的话
                )
                # print(f"outputs: {outputs}")
                # Extract the speech tokens
                generated_ids = outputs[0][input_ids.shape[1] : -1]
                # print(generated_ids.shape)
                print(f"generated_ids: {generated_ids}")
                speech_tokens = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                # print(f"speech_tokens1: {speech_tokens}")

                # Convert  token <|s_23456|> to int 23456
                speech_tokens = extract_speech_ids(speech_tokens)
                # print(f"speech_tokens2: {speech_tokens}")

                speech_tokens = (torch.tensor(speech_tokens)).cuda().unsqueeze(0)
                # print(f"speech_tokens3: {speech_tokens}")

                audio_hat = audio_decode_cosyvoice(
                    speech_tokens,
                    args.input_text,
                    args.prompt_text,
                    args.prompt_speech_path,
                    token2wav_model,
                )

                audio = audio_hat.squeeze(0).cpu().numpy()
                save_file = f"gen_idx_{ix}_text_{i}.wav"
                # print(f"Saving generated audio to {save_file}")
                sf.write(save_file, audio, 24000)
                duration = len(audio) / 24000
                # print(f"Generated audio duration: {duration} seconds")

                save_list.append([save_file, duration])

        for item in save_list:
            print(f"File: {item[0]}, Duration: {item[1]} seconds")
