
Text-to-Image generation using Stable Diffusion


The code utilizes Stable Diffusion models to generate high-quality, photorealistic images from text prompts. It loads a pretrained model, GHArt/SDXL_Yamer_s_Realistic_V4v_xl_fp16, and integrates LoRA (Low-Rank Adaptation) weights to fine-tune the image generation. The script specifies a detailed prompt describing the desired image, adjusts various parameters such as inference steps and guidance scale, and applies a negative prompt to filter out unwanted features. The generated image is produced using GPU acceleration for faster processing.


## Authors

- [@Haseeb-CS](https://github.com/Haseeb-CS)


## Features

- Generates photorealistic images from text prompts using Stable Diffusion
- Loads a pretrained model (GHArt/SDXL_Yamer_s_Realistic_V4v_xl_fp16) for high-quality image synthesis
- Integrates LoRA (Low-Rank Adaptation) weights for model fine-tuning
- Utilizes GPU acceleration (CUDA) for faster image generation
- Allows customization of image generation parameters, including inference steps and guidance scale
- Applies negative prompts to filter out unwanted features and improve output quality
- Disables the safety checker to remove content filtering restrictions
- Supports half-precision (float16) to optimize memory usage and improve performance

## 🚀 About Me
I'm a Machine Learning Engineer specializing in computer vision, natural language processing (NLP), and image generation. I develop AI solutions that leverage my expertise in these domains to solve complex problems and create innovative applications.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haseeb-CS?tab=repositories)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/shahhaseeb281)



## Installation

Follow these steps to set up and use the code for generating images from text using Stable Diffusion models and LoRA fine-tuning:

Set Up Your Environment: Ensure you have Python 3.x installed. It's recommended to create a virtual environment to manage dependencies:

```
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```
Install Required Libraries: Install the necessary libraries using pip:

```
pip install diffusers[torch] transformers
pip install accelerate peft
pip install git+https://github.com/huggingface/diffusers
pip install --upgrade diffusers
pip install -U peft transformers
```
Configure Your Environment for GPU Usage: Make sure you have CUDA installed and properly configured to utilize GPU acceleration for faster image generation. Refer to the PyTorch installation guide for specific instructions based on your system.

Using the Code: To generate images using any pretrained model and LoRA weights, follow these steps:

Select a Pretrained Model: Replace "GHArt/SDXL_Yamer_s_Realistic_V4v_xl_fp16" with any pretrained model available on the Hugging Face Model Hub.

Set Up the LoRA Weights: Download or specify the path to the LoRA weights file you wish to use and assign it to the lora_path variable.

Run the Code: Copy and paste the code into your Python environment and execute it. The script will generate an image based on the specified prompt, LoRA weights, and other parameters:

```
from diffusers import AutoPipelineForText2Image
import torch

# Load the Stable Diffusion model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "your_pretrained_model_here", torch_dtype=torch.float16
).to("cuda")
pipeline.safety_checker = None

# Integrate LoRA weights
lora_path = "path_to_your_lora_weights"
pipeline.load_lora_weights(lora_path)
lora_scale = 0.7

# Define prompt and parameters
prompt = "Your descriptive prompt here"
steps = 40
guidance = 9
neg = "List of negative prompts to filter undesired features"

# Generate the image
image = pipeline(
    prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    negative_prompt=neg,
    # cross_attention_kwargs={"scale": lora_scale}
).images[0]

# Display the generated image
image.show()
```
Customize Your Image Generation:

Modify the prompt variable with the desired text description for the image.
Adjust num_inference_steps, guidance_scale, and lora_scale to fine-tune the output quality.
Use negative_prompt to filter out unwanted elements in the generated image.

# Additional Notes:

The code is capable of using any pretrained Stable Diffusion model and LoRA weights to fine-tune the image generation according to specific requirements.
Make sure to have sufficient GPU memory for generating high-resolution images.
Review the Hugging Face documentation for more details on using different models and LoRA weights.