# Create project

```sh
conda create -n transcriber python=3.14.4 -c conda-forge
```

# Setup Transcriber

## Step 1

Install ffmpeg
```sh
brew install ffmpeg
```

Install the dependencies
```sh
conda env create -f environment.yml
```

## Step 2
Get your Hugging Face token

- Go to https://huggingface.co/join and create a free account
- Go and click Agree to accept terms
    - https://huggingface.co/pyannote/speaker-diarization-3.1
    - https://huggingface.co/pyannote/speaker-diarization-community-1
    - https://huggingface.co/pyannote/segmentation-3.0
- Go to https://huggingface.co/settings/tokens → click New token → copy it

# Setup Summarize

install ollama

```sh
brew install ollama
```

Download a model (good balance of quality vs speed on 18GB RAM)

```sh
# Use the llm that you want

ollama pull llama3.1:8b

ollama pull gemma3:12b
```

# Dependencies

if you install some dependencies, run

```sh
conda env export > environment.yml
```