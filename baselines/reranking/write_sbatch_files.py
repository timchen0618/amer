
retriever_list = ["llama-1b_qampari", "llama-3b_qampari", "llama-8b_qampari", "qwen3-4b_qampari", "llama-1b_ambignq", "llama-3b_ambignq", "llama-8b_ambignq", "qwen3-4b_ambignq"]
# retriever_list = ["llama-1b_qampari_multi", "llama-3b_qampari_multi", "llama-8b_qampari_multi", "qwen3-4b_qampari_multi", "llama-1b_ambignq_multi", "llama-3b_ambignq_multi", "llama-8b_ambignq_multi", "qwen3-4b_ambignq_multi"]
# command = "reranking" # gen doc similarity
command = "mmr"       # actually perform mmr

if command == "reranking":
    template = open("../run_reranking.SBATCH", "r").read()

    for base_retriever in retriever_list:
        with open(f"../sbatch_jobs/run_reranking_{base_retriever}.SBATCH", "w") as f:
            f.write(template.replace("[base_retriever]", base_retriever))
            
    fw = open("../submit.sh", "w")
    for base_retriever in retriever_list:
        fw.write(f"sbatch sbatch_jobs/run_reranking_{base_retriever}.SBATCH\n")
    fw.close()
    
if command == "mmr":
    template = open("../mmr.SBATCH", "r").read()

    for base_retriever in retriever_list:
        for lambda_val in [0.9, 0.75, 0.5]:
            with open(f"../sbatch_jobs/mmr_{base_retriever}_{lambda_val}.SBATCH", "w") as f:
                f.write(template.replace("[base_retriever]", base_retriever).replace("[lambda]", str(lambda_val)))
            
    fw = open("../submit.sh", "w")
    for base_retriever in retriever_list:
        for lambda_val in [0.9, 0.75, 0.5]:
            fw.write(f"sbatch sbatch_jobs/mmr_{base_retriever}_{lambda_val}.SBATCH\n")
    fw.close()
