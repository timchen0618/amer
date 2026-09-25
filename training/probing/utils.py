import wandb
import torch

def train_loop(model, dataloader, eval_dataloader, criterion, optimizer, evaluate_func, metric='accuracy', scheduler=None, device='cpu', epochs=30):
    best_metric = 0 if metric == 'accuracy' else 100000
    model_state_dict = None
    
    model.train()
    for epoch in range(epochs):
        losses = []
        for example in dataloader:
            _input = torch.tensor(example['embeddings']).to(device)
            labels = torch.tensor(example['labels']).to(device)
            if metric == 'mse':
                labels = labels.float()
            optimizer.zero_grad()
            outputs = model(_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        
        
        out_metric = evaluate_func(model, eval_dataloader, device)
        if scheduler is not None:
            scheduler.step(out_metric)
        _lr = optimizer.param_groups[0]['lr']
        print_string = f'Epoch {epoch + 1} | Loss: {sum(losses)/len(losses)} | {metric}: {out_metric:.2f}%' if scheduler is None else f'Epoch {epoch + 1} | Loss: {sum(losses)/len(losses)} | {metric}: {out_metric:.2f}% | LR: {_lr}'
        print(print_string)
        
        if metric == 'accuracy':
            if out_metric > best_metric:
                best_metric = out_metric
                model_state_dict = model.state_dict()
        elif metric == 'mse':
            if out_metric < best_metric:
                best_metric = out_metric
                model_state_dict = model.state_dict()
        else:
            raise NotImplementedError
        
        wandb.log({metric: out_metric, "loss": sum(losses)/len(losses), "epoch": epoch + 1})
    return best_metric, model_state_dict
        
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    
def save_state_dict(state_dict, model_path):
    torch.save(state_dict, model_path)