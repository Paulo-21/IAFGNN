from kan import KAN
import torch
import DatasetLinear

task = "DC-CO"
device = "cpu"
instances , labels = DatasetLinear.get_dataset_kan(task=task, device=device)
instances_test , labels_test = DatasetLinear.get_dataset_kan_test(task=task, device=device)
dataset = {}
print(len(instances))
print(len(labels))
dataset['train_input'] = instances
dataset['test_input'] = instances_test
dataset['train_label'] = labels
dataset['test_label'] = labels_test
print(dataset['train_input'])
print(dataset['train_label'])
print("-------------------------------------")


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

model = KAN(width=[8,5,2], grid=3, k=3, device=device)

results = model.train(dataset, opt="LBFGS", steps=10, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss());

lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
model.auto_symbolic(lib=lib)

formula1, formula2 = model.symbolic_formula()[0]
formula1
formula2
#results['train_acc'][-1], results['test_acc'][-1]
