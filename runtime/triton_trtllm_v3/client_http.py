# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import requests
import soundfile as sf
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="/gpfs01/nfs_share/data20250106/panxb/waihu_yixin/cosyvoice_trtllm/CosyVoice/asset/zero_shot_prompt_16000.wav",
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="希望你以后能够做的比我还好呦。",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="cosyvoice3",
        choices=["f5_tts", "spark_tts", "cosyvoice3"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output_v3_try3.wav",
        help="Path to save the output audio",
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{args.model_name}/infer"
    print(f"url: {url}")
    waveform, sr = sf.read(args.reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"
    print(f"waveform:{waveform}")
    samples = np.array(waveform, dtype=np.float32)
    print(f"samples:{samples}")
    data = prepare_request(samples, args.reference_text, args.target_text)

    import time

    for idx in range(10):
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
        print(result["outputs"][0].keys())
        audio = result["outputs"][0]["data"]
        print(f"audio len: {len(audio)}")
        audio = np.array(audio, dtype=np.float32)
        if args.model_name == "spark_tts":
            sample_rate = 16000
        else:
            sample_rate = 24000
        # save_file = args.output_audio
        save_file = f"output_v3_idx{idx}.wav"
        print(f"Saving generated audio to {save_file}")
        sf.write(save_file, audio, sample_rate, "PCM_16")
        duration = len(audio) / sample_rate
        rtf = (end_time - st_time) / duration
        print(
            f"Total time: {end_time - st_time:.3f}s, Audio duration: {duration:.3f}s, RTF: {rtf:.3f}"
        )
