# Testing LLAMA

Various tutorials are located [here](https://github.com/meta-llama/llama-recipes/tree/main).

Other doucmentation [here](https://llama.meta.com/docs/overview/).

## Download models

Git clone the Meta [Llama 3 repo](https://github.com/meta-llama/llama3). 
Run the download.sh script and follow the instructions. This will download the model checkpoints and tokenizer.

To get the URL required for the download script, request a link from the [Meta Llama team](https://llama.meta.com/).

We will assume the llama checkpoints are downloaded to the directory ``${LLAMA_DIR}``.

```
git clone https://github.com/meta-llama/llama-models.git
cd llama-models/models/llama3_1
export LLAMA_DIR=${PWD}
bash download.sh
```

Downloads will make you choose a model. For the 8B model, we set the following variables:
```bash
export LLAMA_MODEL_DIR=${LLAMA_DIR}/Meta-Llama-3.1-8B
export LLAMA_MODEL_SIZE=8B
```

## Dependencies

```bash
pip install transformers
pip install accelerate
```

```bash
cd ${LLAMA_DIR}
python3 -m venv hf-convertor
source ${LLAMA_DIR}/hf-convertor/bin/activate
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
pip install torch tiktoken blobfile accelerate
```

## Convert Model Weights

```bash
source ${LLAMA_DIR}/hf-convertor/bin/activate
python3  ${LLAMA_DIR}/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir ${LLAMA_MODEL_DIR} \
--output_dir ./models/${LLAMA_MODEL_SIZE} \
--model_size ${LLAMA_MODEL_SIZE}
```