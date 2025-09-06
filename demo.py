import os
import sys
import argparse
import soundfile as sf
import numpy as np
import inference.echox_stream as Echox

os.environ["TOKENIZERS_PARALLELISM"] = "false"
inference_model = Echox.EchoxAssistant()

def process_audio_text(text_prompt, audio_path, output_path):
    tmp = [{
        "conversations": [
            {
                "from": "user",
                "value": text_prompt,
                "audio": audio_path
            }
        ]
    }]

    accumulated_text = ""
    audio_segments = []

    try:
        for text_response, audio_data in inference_model._inference(tmp):
            if text_response:
                new_text = text_response[len(accumulated_text):]
                if new_text:
                    print(new_text, end='', flush=True)
                accumulated_text = text_response

            if audio_data is not None:
                sr, audio = audio_data
                audio_segments.append(audio)

        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_path, final_audio, samplerate=sr)
            print(f"\nAudio saved to: {output_path}")
        else:
            print("No audio generated.")

    except Exception as e:
        print(f"Error: {e}")
        return None

    return accumulated_text

def main():
    text = ""
    audio = "./samples/2.wav"
    output = "./output_response.wav"
    result = process_audio_text(
        text_prompt=text,
        audio_path=audio,
        output_path=output
    )
    print("\n\nResult: " + result)

if __name__ == "__main__":
    main()