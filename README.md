# FTDataGen

Use LLM to generate training data for fine-tuning LLM

## Steps:
1. Parse input file (pdf, docx, txt, etc.)
2. Generate questions and answers by LLM
3. Save data to jsonl file

### output data format: jsonl 
```txt
{"instruction": "instruction", "output": "output"}
{"instruction": "instruction", "output": "output"}
```

## Get Started
### 1. Build docker image
```bash
# for CPU
docker build -t ft-data-gen:cpu .
# for GPU
docker build -t ft-data-gen:gpu .
```
### 2. Run docker container
```bash
# for CPU
docker run -it --rm -v ${PWD}:/workspace ft-data-gen:cpu bash
# for GPU
docker run -it --rm -v ${PWD}:/workspace --gpus all ft-data-gen:gpu
```

### 3. Setup environment variables
```bash
cp .env.example .env
```
Modify `.env` file with your own LLM model and API key
Here we use litellm to support multiple LLM models, you can refer to [litellm](https://docs.litellm.ai/docs/providers) for more details.
##### Ollama example:
```bash
LLM_MODEL=ollama/llama3.1:70b
LLM_BASE_URL=http://localhost:11434
```
##### OpenAI example:
```bash
LLM_MODEL=openai/gpt-4o
LLM_API_KEY=sk-proj-.....
```

### 4. Generate data
Arguments:
- `--input_file`: input file path
- `--qa_num`: number of questions
- `--output_folder`: output folder

```bash
python generate_data.py --input_file data/test.txt --qa_num 2 --output_folder output
```

### 5. Find the output data in `output` folder, `output/training_data.jsonl` is the final training data