from datasets import concatenate_datasets, load_from_disk
import shutil

for retriever in ['inf']:
    for data_name in ['qampari', 'ambiguous_qe']:
        for base_model_name in ['llama-1b', 'llama-3b', 'llama-8b', 'qwen3-4b']:
            for split in ['train']:
                start_and_end_map = {"qampari": [5, 9], "ambiguous_qe": [2, 6]}
                dataset_paths = [f'../training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_{i}_ctxs' \
                for i in range(start_and_end_map[data_name][0], start_and_end_map[data_name][1])]
                assert len(dataset_paths) == 4
                dataset_paths = [load_from_disk(path) for path in dataset_paths]

                combined_dataset = concatenate_datasets(dataset_paths)
                combined_dataset.save_to_disk(f'../training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_{start_and_end_map[data_name][0]}_to_{start_and_end_map[data_name][1]-1}_ctxs')

                # remove the datasets of individual length (directories), even if they are not empty
                for i in range(start_and_end_map[data_name][0], start_and_end_map[data_name][1]):
                    shutil.rmtree(f'../training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_{i}_ctxs')