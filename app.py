import gradio as gr
import os
import inference.echox_stream as Echox
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":

    inference_model = Echox.EchoxAssistant()

    def process_audio_text(text, audio):
        audio_path = audio
        # text = ""
        tmp = [{
            "conversations":[
                {
                    "from": "user",
                    "value": text,
                    "audio": audio_path
                }
            ]
        }]
        
        accumulated_text = ""
        
        for text_response, audio_data in inference_model._inference(tmp):
            if text_response:
                accumulated_text = text_response
            
            if audio_data is not None:
                sr, audio = audio_data
                yield (sr, audio), accumulated_text
            else:
                yield None, accumulated_text

    examples = [
        ["Recognize what the voice said and respond to it.", "./samples/1.wav"],
        ["", "./samples/2.wav"],
    ]

    iface = gr.Interface(
        fn=process_audio_text,
        inputs=[
            gr.Textbox(label="Input Text", value=examples[0][0]),
            gr.Audio(type="filepath", label="Upload Audio", value=examples[0][1])
        ],
        outputs=[
            gr.Audio(label="Streamed Audio", streaming=True, autoplay=True),
            gr.Textbox(label="Model output")
        ],
        examples=examples,
        live=False,
        allow_flagging="never"
    )

    iface.launch(server_name="0.0.0.0", server_port=7860)