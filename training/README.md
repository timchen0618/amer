# Training (Contriever)
## How to run training 
The following script should work once you specify the `data_path` to be the training data, `OUTPUT_DIR` to be where you store the checkpoints and `run_name` to be the name of the wandb experiment. 
You should set these arguments to values that you would remember later (for better organization).

```
    accelerate launch --main_process_port $PORT finetuning.py --train_data ${data_path}  \
                                                    --eval_data ${data_path}  \
                                                    --temperature 0.05 \
                                                    --total_steps 50000 \
                                                    --warmup_steps 1000 \
                                                    --lr 0.0001 \
                                                    --save_freq 2500 \
                                                    --log_freq 250 \
                                                    --eval_freq 2500 \
                                                    --negative_hard_ratio 0.0 \
                                                    --negative_ctxs 1 \
                                                    --per_gpu_batch_size 48 \
                                                    --per_gpu_eval_batch_size 48 \
                                                    --model_path facebook/contriever-msmarco \
                                                    --chunk_length 512 \
                                                    --accumulation_steps 2 \
                                                    --run_name ${run_name} \
                                                    --output_dir ${OUTPUT_DIR} \
                                                    --training_mode linear_projection \
                                                    --max_positive_documents 1 
```


## Codebase Structure
### Training `contriever/` (see `contriever/` folder)
#### Python Scripts
- `evaluate_retrieved_passages.py`:
- `finetuning.py`: the main training script. Use `accelerate` package to handle multi-GPU training. 
- `generate_passage_embeddings.py`: generate embeddings for all the documents (passages) in the corpus before doing retrieval. This has to be done once for all the model after it has been trained. 
- `passage_retrieval.py`: 
- `src`: a lot of different source scripts taken from "contriever" original repository. Only explain important ones. 
    - `contriever.py`: Mainly modified the `load_retriever()` function to handle newly defined inbatch classes defined in `inbatch.py`. 
    - `finetuning_data.py`: define the "Dataset" class that helps load the data. Added functionality of handling different data format for different model archietecture and varying number of input documents. 
    - `inbatch.py`: defines how models of different architectures take in query / input documents. 
    - `option.py`: hold the argument parser and tell you the valid arguments to pass to the script. 

#### training scripts
- `run.sh`: A working example for training. 
- `run_sample.sh`: A working example for training, but we sample the number of input documents for each query. (This has not worked very well.)
- `hypertune.sh`: Hyperparameter tuning version for `run.sh`. 

#### inference scripts
- `gen_embed.sh`: generate passage embeddings for all the documents in the corpus. 
- `retrieve.sh`: actually doing retrieval. Specify the mode the model is trained on using `--mode [mode]`. Use `standard` mode for the base retriever. 



### Data Processing
- `process_qampari_data.py` & `process_qampari_data.ipynb`: some processing function for making the QAMPARI data. Find the gold documents (that provide each answer to the question) within the Wikipedia corpus. 
- `get_training_data_qampari.ipynb`: obtain data for training diverse retrievers on QAMPARI from the matched data we got using the previous script. This would include "rewritten question", "positive documents", and "negative documents". 
- `get_data_quest.py`: obtain data for training diverse retrievers on QUEST. 

### Probing Answer Length (Predict number of answers)
- `probing/`: probing to see if we could predict answer length (number of answers) based on the question alone. 
