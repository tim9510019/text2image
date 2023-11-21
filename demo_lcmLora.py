from diffusers import DiffusionPipeline, LCMScheduler
from compel import Compel, ReturnedEmbeddingsType
import torch
import os
import gradio as gr
import psutil

# HuggingFace avoid harmful content
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None)
# Support NVIDAI RTX / V100 / P100 / A100 / H100
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# Check if MPS is available on OSX only M1/M2/M3 chips (APPLE M series)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "xpu" if xpu_available else "cpu"
)
torch_device = device
torch_dtype = torch.float16

print(f"SAFETY_CHECKER: {SAFETY_CHECKER}")
print(f"TORCH_COMPILE: {TORCH_COMPILE}")
print(f"device: {device}")

if mps_available:
    device = torch.device("mps")
    torch_device = "cpu"
    torch_dtype = torch.float32

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

if SAFETY_CHECKER == "True":
    pipe = DiffusionPipeline.from_pretrained(model_id)
else:
    pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device=torch_device, dtype=torch_dtype).to(device)
pipe.unet.to(memory_format=torch.channels_last)

# check if computer has less than 64GB of RAM using sys or os
if psutil.virtual_memory().total < 64 * 1024**3:
    pipe.enable_attention_slicing()

if TORCH_COMPILE:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

    pipe(prompt="warmup", num_inference_steps=1, guidance_scale=8.0)

# Load LCM LoRA
pipe.load_lora_weights(
    "latent-consistency/lcm-lora-sdxl",
    use_auth_token=HF_TOKEN,
)

compel_proc = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)


def predict(
    prompt, guidance, steps, seed=1231231, progress=gr.Progress(track_tqdm=True)
):
    generator = torch.manual_seed(seed)
    prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)

    results = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=1024,
        height=1024,
        # original_inference_steps=params.lcm_steps,
        output_type="pil",
    )
    nsfw_content_detected = (
        results.nsfw_content_detected[0]
        if "nsfw_content_detected" in results
        else False
    )
    if nsfw_content_detected:
        raise gr.Error("NSFW content detected.")
    return results.images[0]


css = """
#container{
    margin: 0 auto;
    max-width: 40rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# SDXL in 4 steps with Latent Consistency LoRAs
            SDXL is loaded with a LCM-LoRA, giving it the super power of doing inference in as little as 4 steps. [Learn more on our blog](https://huggingface.co/blog/lcm_lora) or [technical report](https://huggingface.co/papers/2311.05556).
            """,
            elem_id="intro",
        )
        with gr.Row():
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Insert your prompt here:", scale=5, container=False
                )
                generate_bt = gr.Button("Generate", scale=1)

        image = gr.Image(type="filepath")
        with gr.Accordion("Advanced options", open=True):
            guidance = gr.Slider(
                label="Guidance", minimum=0.0, maximum=5, value=0.3, step=0.001
            )
            steps = gr.Slider(label="Steps", value=4, minimum=2, maximum=10, step=1)
            seed = gr.Slider(
                randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1
            )

        inputs = [prompt, guidance, steps, seed]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image)

demo.queue()
demo.launch()
