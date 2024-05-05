import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import statistics
import DatasetLinear

af_data_root = "../af_dataset/"
result_root = "../af_dataset/all_result/"
task = "DC-CO"
print(task)
MAX_ARG = 200000
v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
print("PYTORCH_CUDA_ALLOC_CONF : ", v)
INPUT_FEATURES = 9
HIDDEN_FEATURES = INPUT_FEATURES*INPUT_FEATURES

class MultiLinear(nn.Module):
    def __init__(self):
        super(MultiLinear, self).__init__()
        self.layer1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.layer2 = nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.layer3 = nn.Linear(HIDDEN_FEATURES, INPUT_FEATURES*2)
        self.layer4 = nn.Linear(INPUT_FEATURES*2, 1)
    def forward(self, inputs):
        h = self.layer1(inputs)
        h = F.leaky_relu(h)
        #h = F.gelu(h)
        #h = F.silu(h)
        #h= F.mish(h)
        h = self.layer2(h)
        h = F.leaky_relu(h)
        #h = F.silu(h)
        #h= F.mish(h)
        h = self.layer3(h)
        h = F.leaky_relu(h)
        #h = F.silu(h)
        #h= F.mish(h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h  # Remove the last dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
model_dir = "model_save/"
model_path = model_dir + "linear_"+task+"_"+str(INPUT_FEATURES)+"f.pth"
print("runtime : ", device)
model = MultiLinear().to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total parameters : ", total_params)
#if os.path.exists(model_path):
    #model.load_state_dict(torch.load(model_path))
    #print("Model as being loaded")
    
loss = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Loading Data...")
#af_dataset = DatasetDGL.TrainingGraphDataset(af_data_root+"dataset_af/", af_data_root+"result/", task=task, device=device)
tic = time.perf_counter()
af_dataset = DatasetLinear.TrainingLinearDataset(task=task, device=device)
print(time.perf_counter()-tic)
print("Start training")
model.train()
for epoch in range(1000):
    tot_loss = []
    tot_loss_v = 0
    for (inputs, label) in af_dataset:
        optimizer.zero_grad()
        out = model(inputs)
        losse = loss(out, label)
        losse.backward()
        optimizer.step()
        tot_loss.append(losse.item())
        tot_loss_v += losse.item()
    print("Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "loss : ", tot_loss_v)

print("final test start")
#TEST the Model
#NORMAL
model.eval()
torch.save(model.state_dict(), model_path)
#SCRIPT
example = torch.rand(INPUT_FEATURES, device=device)
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("model_ln/model_ln"+task+"_"+INPUT_FEATURES+".pt")
#ONNX
task_f = task.replace("-", "_")
onnx_path = "model_ln/linear_"+task_f+"_"+str(INPUT_FEATURES)+"f.onnx"
#onnx_program = torch.onnx.dynamo_export(model, example)
torch.onnx.export(model,               # model being run
                  example,                         # model input (or a tuple for multiple inputs)
                  onnx_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  #opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                )
print(onnx_path)
#onnx_program.save(onnx_path)
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)