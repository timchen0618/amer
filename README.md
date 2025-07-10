# Training an Autoregressive Retriever


## Data Creation
### Generation Procedure
#### Retrieve Documents for Questions 
For NQ, MSMARCO, QAMPARI, and AmbigNQ, you should first retrieve documents for the questions, and then find the positive / negative documents. To find the positive documents, either match the documents to the answers or see if it is in the gold documents set (match using document indices). 
#### Generate Embeddings for the Documents 
For NQ, MSMARCO, QAMPARI, and AmbigNQ, after you have retrieved the documents and identified the positive / negative documents, you could generate embeddings for the documents. You should first generate document embeddings for a specific number of positive documents, and later combine them.  
For example, for QAMPARI, you would generate embeddings for number of positive documets = [5,6,7,8], and later combine them into a single set (N=30k). 
#### Generate Document Indices and Attention Mask
Use `data_creation/create_input_for_contriever.py` to create the document indices and attention mask, and put the positive / negative documents embeddings to the right place. For the first two steps, use `gen_embed_for_auto_training.py` in `Multi_Answer/mteb_retriever/`. 

### Data Format
#### Data Format for Training 
{
    `"input_ids"`: the input indices (of the query) tokenized by the Llama tokenizer, 
    `"attention_mask"`: mask indicating which tokens are used, 
    `"positive_embeddings"`: (optional) only exist when using Contrastive Loss. The embeddings of positive documents. Size: (length, dim), 
    `"negative_embeddings"`: (optional) only exist when using Contrastive Loss. The embeddings of negative documents. Size: (length, dim), 
    `"labels"`: (optional) only exist when using MSE Loss. The embeddings of the ground truth vectors. Size: (length, dim). 
}
#### Data Format for Generation
Could use `input_ids`, or use the raw text to do the generation. 


### Path to Data 
- `data/`: contains the original data structure of different datasets. 
- `training_datasets/`: training sets 
    - structure: [dataset]/[retriever]/[actual_dataset]
- `data_creation/raw_data/`: 
    - `[dataset]_[split]_question_only.jsonl`: the JSONL files that contain only the questions. For generating the data. 
    - `[dataset]_[retriever]/`: the `.npy` file for embeddings of the positive / random negative / hard negative documents. 
- `data_creation/gaussian/`: contains the code needed for generating gaussian data. 

## Training 
The training is done using LoRA using a Llama-1B base model. We train the model using either a single GPU or multiple GPUs in order to use a larger batch size. The script is written using `pytorch / huggingface / accelerate` packages.
### Run Training 
We support either single-GPU or multi-GPU training.  
#### Single-GPU
Run command `python train.py`. The configurations are read from `configs/` folder. Use the config files correspondingly. 
#### Multi-GPU
With multiple-GPU training, the script is written using the `accelerate` package. 

### Custom Dataset
Modify `dataset.py` to include new data collators that could process different types of data.  
`dataset.py` also contains some useful processing functions.  

### Models 
`model.py` contains the definition of the embedding model, and the loading logic of the model.  
It also contains different loss functions, including `MSELoss`, `HungarianMSELoss`, `ContrastiveLoss`, and `HungarianContrastiveLoss`.  
The `EmbeddingModel` class includes three submodules: `base_causallm` (the base casaul langauge model), `input_projection` (the projection matrix that maps the embeddings to base LMs' embeddings space), and `output_projection` (the projection matrix that maps the LMs' prediction into retrievers' embedding space).  
Call the `.forward()` function to do training, and call the `.generate()` function for prediction.  

### Misc. 
- `dist_utils.py`: contains functions that handle multiple-GPU training. 
- `eval_utils.py`: contains functions that help the evaluation. 
- `retrieval_utils.py`: contains classes / functions that helps the retrieval (e.g. Indexer).
- `utils.py`: learning rate scheduler, optimizer, etc. 


## Evaluation 
First run `gen_ret_and_eval.py` to generate the query vectors and retrieve documents using the query vectors.  
Then run `eval.py` to evaluate the performances of the generated document set.  
See `scripts/eval.sh` for an example to run the full pipeline.  

### Usage Examples of `gen_ret_and_eval.py`

#### Basic Usage
```bash
python gen_ret_and_eval.py
```

#### Different Dataset and Retriever
```bash
python gen_ret_and_eval.py \
    --data_name nq \
    --training_data_name nq \
    --retriever_list stella inf \
    --suffix_list "_contrastive_lr2e5_ep20_temp0.05_warmup0.05"
```
#### Google API Usage
```bash
python gen_ret_and_eval.py \
    --data_name arguana_generated \
    --google_api \
    --retriever_list inf
```

#### Custom Checkpoint Number
```bash
python gen_ret_and_eval.py \
    --data_name ambiguous_qe \
    --checkpoint_num 2500 \
    --use_suffix_mapping false
```

### Usage example of eval.py
You should specify the `--data-type`, which tells the script which ground truth data to use.  
You should also specify the `--root`, which tells the script where to look for the prediction file.  
`--topk` tells the script which k it sohuld compute metrics @ k.  
For example:  
```bash
python eval.py --data-type qampari \
    --root /path/to/prediction \
    --topk 10 100 --has-gold-id
```

If you were evaluating on NQ or MSMARCO for stage 1, you could use `eval_nq_msmarco.py`.  


## How to Submit Jobs
**Submitting Jobs to SLURM**  
If you are using SLURM, use `hyperparameter_search.sh` to submit jobs (to HPC).  
The configs files are in `sbatch_configs/`. (For AmbigNQ and for QAMPARI)  
Or you could simply follow `scripts/run.SBATCH` to just run `train.py`.  


**Submitting Interactive Jobs**  
If you are using an interactive machine, use `scripts/single_run.sh` to run a job. 


### Example scripts

#### Run Training 
Use `scripts/run_dist.SBATCH` for distributed training and `scripts/run_single.SBATCH`.  


#### Run Eval 
Use `scripts/submit_eval_jobs.sh` to submit jobs to SLURM for eval.  