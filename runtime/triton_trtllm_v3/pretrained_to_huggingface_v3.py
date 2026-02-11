import sys
sys.path.insert(0, '/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice')
sys.path.insert(0, '/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
from argparse import ArgumentParser
from transformers import AutoTokenizer
import torch
from loguru import logger


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--pretrained-cosyvoice3-path",
        type=str,
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/Fun-CosyVoice3-0.5B-2512",
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default='/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/runtime/triton_trtllm_v3/transformers_cosyvoice3_llm',
        help="The path to save the model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    cosy3_model = AutoModel(
        model_dir=args.pretrained_cosyvoice3_path, load_vllm=False, load_trt=False, fp16=False
    )

    llm = cosy3_model.model.llm.llm.model

    speech_embedding = cosy3_model.model.llm.speech_embedding
    llm_decoder = cosy3_model.model.llm.llm_decoder
    # 没有这玩意了
    # llm_embedding = cosy3_model.model.llm.llm_embedding 

    tokenizer = AutoTokenizer.from_pretrained(f"{args.pretrained_cosyvoice3_path}/CosyVoice-BlankEN")
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    # original_tokenizer_vocab_size=151663
    original_tokenizer_vocab_size = len(tokenizer)
    # https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/file/view/master/cosyvoice3.yaml?status=1#L26
    cosyvoice_token_size = 6561
    new_tokens = [f"<|s_{i}|>" for i in range(cosyvoice_token_size)] + [
        "<|eos1|>", "<|eos2|>", "<|eos3|>", "<|sos|>", "<|task_id|>"
    ]
    logger.info(f"new_tokens: {new_tokens}")
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=256)
    vocab_size = llm.get_input_embeddings().weight.shape[0]

    feature_size = speech_embedding.embedding_dim
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=vocab_size, bias=False)

    with torch.no_grad():
        # set the weight and bias of the new lm_head to 0
        new_lm_head.weight.data.zero_()
        # new_lm_head.bias.data.zero_()
        # 甭+3了，因为llm_decoder维度在cosy3变了, +200是为了维度一致...至于为什么俺也不知, 而bias整个为None了
        new_lm_head.weight[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice_token_size+200] = llm_decoder.weight
        # new_lm_head.bias[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice_token_size] = llm_decoder.bias

    llm.lm_head = new_lm_head
    input_embeddings = llm.get_input_embeddings()

    with torch.no_grad():
        input_embeddings.weight[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice_token_size+200] = speech_embedding.weight
        # input_embeddings.weight[original_tokenizer_vocab_size+cosyvoice_token_size+3:original_tokenizer_vocab_size+cosyvoice_token_size+3+2] = llm_embedding.weight

    eos_token_ids = [original_tokenizer_vocab_size + cosyvoice_token_size, original_tokenizer_vocab_size + cosyvoice_token_size + 1, original_tokenizer_vocab_size + cosyvoice_token_size + 2]
    llm.generation_config.eos_token_id = eos_token_ids
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 0.8
    llm.generation_config.top_k = 25

    llm.config.eos_token_id = original_tokenizer_vocab_size + cosyvoice_token_size
    llm.config.vocab_size = vocab_size
    llm.config.tie_word_embeddings = False
    llm.config.use_bias = False
    llm.to(torch.bfloat16)
    llm.save_pretrained(args.save_path)

    TEMPLATE = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- '<|sos|>' + message['content'] + '<|task_id|>' }}{%- elif message['role'] == 'assistant' %}{{- message['content']}}{%- endif %}{%- endfor %}"
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(args.save_path)


# sos=6561, eos_token=6562, task_id=6563, fill_token=6564
