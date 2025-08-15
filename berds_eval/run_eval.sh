#!/bin/bash
for corpus_and_retriever in google_api,Contriever
do
    IFS=","
    set -- ${corpus_and_retriever}
    echo "corpus: $1 | retriever: $2"
    #ROOT="/scratch/cluster/hungting/projects/Multi_Answer/Data/retrieval_outputs/${1}/${2}"
    #ROOT="/datastor1/hungting/retrieval_outputs/$2/$1"
    #ROOT="/datastor1/hungting/retrieval_outputs/mteb_retriever/$2/$1"
    ROOT="/scratch/cluster/hungting/projects/autoregressive/results/berds_inf/toy_contrastive_from_stage2_lr1e4_ep30_temp0.05_warmup0.05_eos_gradnorm1/"
    PORT=29500
    TOPK=100

    #for DATA in "arguana_generated_1k_sorted.test.jsonl.mistralpred" "kialo_1k_sorted.test.jsonl.mistralpred" "opinionqa_1k_sorted.test.jsonl.mistralpred"
    for DATA in "retrieval_out_dev_arguana_generated.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_single.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_from_2nd_to_3rd.jsonl.mistralpred"
    # for DATA in "retrieval_out_dev_arguana_generated.jsonl" "retrieval_out_dev_arguana_generated_single.jsonl" "retrieval_out_dev_arguana_generated_from_2nd_to_3rd.jsonl"
    #for DATA in  "arguana_generated_1k.jsonl" "opinionqa_1k.jsonl" "kialo_1k.jsonl"
    #for DATA in "opinionqa_1k_sorted.test.jsonl"
    do
        MODEL_SHORT="mistral"
        MODEL_NAME="timchen0618/Mistral_BERDS_evaluator_full"
        OUTPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"

        PYTHONPATH=.. python eval_vllm.py \
                --data ${ROOT}/${DATA} \
                --output_file ${OUTPUT}   \
                --instructions instructions_chat.txt \
                --model ${MODEL_NAME}  \
                --model_type ${MODEL_SHORT} \
                --topk ${TOPK} \
                --compute_only
    done
done








