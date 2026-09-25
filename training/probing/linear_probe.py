import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np

import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torch.optim.lr_scheduler

from data import load_data
from models import LinearProbe, LinearRegressionProbe, MLPProbe, MLPRegressionProbe
from utils import train_loop, save_state_dict
        
# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for example in dataloader:
            _input = torch.tensor(example['embeddings']).to(device)
            labels = torch.tensor(example['labels']).to(device)
            outputs = model(_input)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            for (p, l) in zip(predicted, labels):
                valid_labels = [l+2, l+1, l, l-1, l-2]
                if p in valid_labels:
                    correct += 1
    accuracy = correct / total
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    return 100*accuracy

# Evaluation function
def evaluate_mse(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataloader:
            _input = torch.tensor(example['embeddings']).to(device)
            labels = torch.tensor(example['labels']).float().to(device)
            outputs = model(_input)
            losses.append(torch.nn.functional.mse_loss(outputs, labels))
            
    return sum(losses) / len(losses)


def main(args):
    # Training setup
    # model = LinearProbe()
    model_collection = {"linear_probe": LinearProbe, "linear_probe_regression": LinearRegressionProbe, "mlp_probe": MLPProbe, "mlp_probe_regression": MLPRegressionProbe}
    print('Using model', args.model_name)
    
    if 'regression' in args.model_name:
        print('doing regression...')
        criterion = nn.MSELoss()
        metric = "mse"
    else:
        print('doing classification...')
        criterion = nn.CrossEntropyLoss()
        metric = "accuracy"
    
    # load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(regression=('regression' in args.model_name))

    best_metric = 0 if metric == 'accuracy' else 100000
    # schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)]
    # for lr in [1e-3, 2e-3, 5e-3, 1e-4, 2e-4, 5e-4, 1e-5, 2e-5, 5e-5]:
    for lr in [5e-4]:
        # for batch_size in [8,16,32,64,128,256]:
        for batch_size in [16]:
            # for num_epoch in [10,20,30,40,50]:
            for num_epoch in [100]:
                
                model = model_collection[args.model_name]()
                model = model.to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                sch_string = 'max' if metric == 'accuracy' else 'min'
                # for num_epoch in [30, 50]:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, sch_string, threshold=0.01)
                
                # scheduler = ()
                for split in ['train', 'dev', 'test']:
                    data[split]['dataloader'] = torch.utils.data.DataLoader(data[split]['dataset'], batch_size=batch_size, shuffle=(split=='train'))
                    
                run = wandb.init(
                    entity="timchen0618",
                    project=args.model_name,
                    name=f"lr{lr}_bsz{batch_size}_schReduce_ep{num_epoch}",
                )
                
                metric_per_run, model_state_dict_per_run = train_loop(model=model, dataloader=data['train']['dataloader'], 
                        eval_dataloader=data['dev']['dataloader'], 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        evaluate_func=evaluate if metric == 'accuracy' else evaluate_mse, 
                        metric=metric, 
                        scheduler=scheduler, 
                        device=device, 
                        epochs=num_epoch)
                run.finish()
                
                if metric == 'accuracy':
                    if metric_per_run > best_metric:
                        print(metric_per_run, best_metric, 'saving model...')
                        best_metric = metric_per_run
                        save_state_dict(model_state_dict_per_run, args.output_dir + f'best_{args.model_name}.pt')
                elif metric == 'mse':
                    if metric_per_run < best_metric:
                        print(metric_per_run, best_metric, 'saving model...')
                        best_metric = metric_per_run
                        save_state_dict(model_state_dict_per_run, args.output_dir + f'best_{args.model_name}.pt')
                else:
                    raise NotImplementedError



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # basic parameters
    parser.add_argument("--output_dir", type=str, default="./checkpoint/", help="models are saved here")
    parser.add_argument("--model_name", type=str, choices=['linear_probe', 'linear_probe_regression', 'mlp_probe', 'mlp_probe_regression'])

    args = parser.parse_args()
        
    main(args)
    
           








