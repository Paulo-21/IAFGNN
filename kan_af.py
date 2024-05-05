from kan import KAN
import torch
import DatasetLinear

torch.set_default_device('cuda')
task = "DC-CO"
device = "cuda"
instances , labels = DatasetLinear.get_dataset_kan(task=task, device=device)
instances_test , labels_test = DatasetLinear.get_dataset_kan_test(task=task, device=device)
dataset = {}
dataset['train_input'] = instances.unsqueeze(dim=1)
dataset['test_input'] = instances_test.unsqueeze(dim=1)
dataset['train_label'] = labels.unsqueeze(dim=1)
dataset['test_label'] = labels_test.unsqueeze(dim=1)
print(dataset['train_input'])
len(dataset['train_input'])
print("-------------------------------------")
model = KAN(width=[9,2,1], grid=3, k=3, device=device)

results = model.train(dataset, opt="Adam", steps=30, loss_fn=torch.nn.MSELoss());
#results['train_acc'][-1], results['test_acc'][-1]
