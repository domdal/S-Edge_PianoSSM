import os
import sys
import yaml
import argparse

import torch
import numpy as np
import pandas as pd


from time import time as get_time
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from src.model.classifier import SC_Model_classifier
from src.utils.GoogleSpeechCommands import SubsetSC
from src.utils.train_test import evaluate, train_one_epoch
from src.utils.LogFile import Echo_STDIO_to_File
from src.utils.augments import augments_weak, augments_strong
from src.utils.experimentManager import ExperimentManagerSaveFunction



model_yaml_path = './model_descriptions/S_Edge_tiny.yaml'
hyperparameters_yaml_path = './hyperparams/Speech_Commands_Default.yaml'



results_path = './results_journal/'

train_history_csv = 'train_history.csv'
train_sub_epoch_history_csv = 'train_sub_epoch_history.csv'


if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

device = 'cuda:0'
num_workers = 0
pin_memory = True


# parser = argparse.ArgumentParser(description='Keyword spotting')
# # Optimizer
# parser.add_argument('--epochs', default=40, type=float, help='Training epochs'),
# # Device
# parser.add_argument('--device', default='cuda:0', type=str,help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
# # Seed
# parser.add_argument('--seed', default=1234, type=int, help='Seed')
# # Dataloader
# parser.add_argument('--num_workers', default=0, type=int,help='Number of workers to use for dataloader')
# parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
# # Model
# parser.add_argument('--input_size', default=1, type=int, help='Input size')
# parser.add_argument('--classes', default=35, type=int, help='number of classes')

# # # # Original S5 Model, ca 300k Params
# # parser.add_argument('--hidden_sizes', default=[64]*6, type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[96]*6, type=list, help='Number of output classes, per layer')
# # parser.add_argument('--zeroOrderHoldRegularization', default=[], type=list, help='Number of output classes, per layer')
# # # parser.add_argument('--zeroOrderHoldRegularization', default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.8,1.0], type=list, help='Number of output classes, per layer')
# # parser.add_argument('--trainable_skip_connections', default=False, type=bool, help='Trainable skip connections', choices=[True, False])
# # parser.add_argument('--lr', default=0.002, type=float, help='Learning rate')

# # # Bittner Model, ca 100k Params
# # parser.add_argument('--hidden_sizes', default=[50]*10, type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[50]*10, type=list, help='Number of output classes, per layer')
# # parser.add_argument('--zeroOrderHoldRegularization', default=[], type=list, help='Number of output classes, per layer')
# # # parser.add_argument('--zeroOrderHoldRegularization', default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.8,1.0], type=list, help='Number of output classes, per layer')
# # parser.add_argument('--trainable_skip_connections', default=False, type=bool, help='Trainable skip connections', choices=[True, False])
# # parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

# #large model2 Run#55
# # parser.add_argument('--hidden_sizes', default=[64,56,48,40,32,24,16,12,8,4], type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[4,8,12,16,24,32,40,48,56,64], type=list, help='Number of output classes, per layer')
# # parser.add_argument('--zeroOrderHoldRegularization', default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.8,1.0], type=list, help='Number of output classes, per layer')
# # parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
# # parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')

# # large model2 Run#55
# parser.add_argument('--hidden_sizes', default=[32, 24, 16, 14, 12, 8], type=list, help='Hidden sizes of the model')
# parser.add_argument('--output_sizes', default=[8, 16, 32, 42, 54, 64], type=list, help='Number of output classes, per layer')
# parser.add_argument('--zeroOrderHoldRegularization', default=[], type=list, help='Number of output classes, per layer')
# parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
# parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')

# # small model 1
# # parser.add_argument('--hidden_sizes', default=[10]*4, type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[10]*4, type=list, help='Number of output classes, per layer')
# # parser.add_argument('--zeroOrderHoldRegularization',default=[], type=list, help='probability that zero order hold is applied')
# # parser.add_argument('--trainable_skip_connections', default=False, type=bool, help='Trainable skip connections', choices=[True, False])
# # parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

# # # small model 2 Run#56
# # parser.add_argument('--hidden_sizes', default=[32, 16, 8], type=list, help='Hidden sizes of the model')
# # # parser.add_argument('--hidden_sizes', default=[16, 64, 128], type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[8, 32, 64], type=list, help='Number of output classes, per layer')
# # # parser.add_argument('--zeroOrderHoldRegularization',default=[0.3,0.6,1.0], type=list, help='probability that zero order hold is applied')
# # parser.add_argument('--zeroOrderHoldRegularization',default=[], type=list, help='probability that zero order hold is applied')
# # parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
# # # parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
# # parser.add_argument('--lr', default=0.002, type=float, help='Learning rate')

# # small model 2 Run#56
# # parser.add_argument('--hidden_sizes', default=[48, 32, 16, 8], type=list, help='Hidden sizes of the model')
# # # parser.add_argument('--hidden_sizes', default=[16, 64, 128], type=list, help='Hidden sizes of the model')
# # parser.add_argument('--output_sizes', default=[16, 32, 64, 96], type=list, help='Number of output classes, per layer')
# # # parser.add_argument('--zeroOrderHoldRegularization',default=[0.25,0.5,0.75,1.0], type=list, help='probability that zero order hold is applied')
# # parser.add_argument('--zeroOrderHoldRegularization',default=[], type=list, help='probability that zero order hold is applied')
# # parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
# # parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
# # # parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')


# parser.add_argument('--input_bias', default=True, type=bool,help='Input bias', choices=[True, False])
# parser.add_argument('--bias_init', default='uniform', type=str,help='Bias initialization', choices=['zero', 'uniform'])
# # parser.add_argument('--bias_init', default='zero', type=str,help='Bias initialization', choices=['zero', 'uniform'])      # Only better if B init modified
# # parser.add_argument('--output_bias', default=False, type=bool,help='Output bias', choices=[True, False])
# parser.add_argument('--output_bias', default=True, type=bool,help='Output bias', choices=[True, False])
# parser.add_argument('--complex_output', default=False,type=bool, help='Complex output', choices=[True, False])
# # parser.add_argument('--norm', default=False, type=bool,help='Normalization', choices=[True, False])
# parser.add_argument('--norm', default=True, type=bool,help='Normalization', choices=[True, False])
# parser.add_argument('--norm_type', default='bn', type=str,help='Normalization type', choices=['ln', 'bn'])
# parser.add_argument('--stability', default='abs', type=str,help='ensure stability', choices=['relu', 'abs'])
# parser.add_argument('--Reduction', default='mean', type=str,help='Reduction', choices=['mean', 'EWMA'])
# # parser.add_argument('--augments', default='strong', type=str, help='Augments', choices=['none', 'weak', 'strong'])
# # parser.add_argument('--augments', default='weak', type=str, help='Augments', choices=['none', 'weak', 'strong'])
# parser.add_argument('--augments', default='none', type=str, help='Augments', choices=['none', 'weak', 'strong'])
# parser.add_argument('--act', default='LeakyRELu', type=str, help='Augments', choices=['RELu', 'LeakyRELu', 'Identity'])
# parser.add_argument('--weight_decay', default=0.04, type=float, help='Weight decay')
# # parser.add_argument('--weight_decay', default=0.00, type=float, help='Weight decay')
# parser.add_argument('--B_C_init', default='S5', type=str, help='Initialization type',choices=['S5','orthogonal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'])
# # parser.add_argument('--B_C_init', default='orthogonal', type=str, help='Initialization type',choices=['S5','orthogonal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'])


# # Parse arguments
# args = parser.parse_args()
# lr = args.lr
# epochs = args.epochs
# device = args.device
# if args.seed == -1:
#     seed = np.random.randint(0, 100000)
# else:
#     seed = args.seed
# num_workers = args.num_workers
# pin_memory = True if (device == 'cuda:0') or (device == 'cuda:1') else False
# batch_size = args.batch_size

# input_size = args.input_size
# classes = args.classes
# hidden_sizes = args.hidden_sizes
# output_sizes = args.output_sizes
# zeroOrderHoldRegularization = args.zeroOrderHoldRegularization
# trainable_skip_connections = args.trainable_skip_connections
# input_bias = args.input_bias
# bias_init = args.bias_init
# output_bias = args.output_bias
# complex_output = args.complex_output
# norm = args.norm
# norm_type = args.norm_type
# B_C_init = args.B_C_init
# stability = args.stability
# reduction = args.Reduction
# augments = args.augments
# act = args.act
# weight_decay= args.weight_decay
# dropout=0.0

with open(model_yaml_path, 'r') as f:
    model_yaml = yaml.safe_load(f)
with open(hyperparameters_yaml_path, 'r') as f:
    hyperparameters_yaml = yaml.safe_load(f)

for key, value in model_yaml.items():   
    globals()[key] = value
for key, value in hyperparameters_yaml.items():
    globals()[key] = value

epochs = 1

# save config dict to file
config_dict = {
    'lr': learning_rate,
    'epochs': epochs,
    'device': device,
    'seed': seed,
    'num_workers': num_workers,
    'batch_size': batch_size,
    'input_size': input_size,
    'classes': classes,
    'hidden_sizes': [hidden_sizes],
    'output_sizes': [output_sizes],
    'zeroOrderHoldRegularization': [zeroOrderHoldRegularization],
    'trainable_skip_connections': trainable_skip_connections,
    'input_bias': input_bias,
    'bias_init': bias_init,
    'output_bias': output_bias,
    'complex_output': complex_output,
    'norm': norm,
    'norm_type': norm_type,
    'B_C_init': B_C_init,
    'stability': stability,
    'augments': augments,
    'act': act,
    'weight_decay' : weight_decay,
    'dropout': dropout,
}

model_save_path = ExperimentManagerSaveFunction(path=results_path, config_dict=config_dict, saveFiles=['src','train_sc.py', model_yaml_path, hyperparameters_yaml_path])

echo_stdio = Echo_STDIO_to_File(os.path.join(model_save_path, 'output.txt'))
sys.stdout = echo_stdio

echo_sterr = Echo_STDIO_to_File(os.path.join(model_save_path, 'error.txt'))
sys.stderr = echo_sterr

print("Echoing to file start")

# Save config dict to file
print("Save config dict to file")
pd.DataFrame.from_dict(config_dict, orient='index').to_csv(os.path.join(model_save_path, 'config.csv'))


device = torch.device(device if torch.cuda.is_available() else "cpu")

# set seed for pytorch and numpy
print(f"Set seed to {seed}")
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
print("Start loading data")
dataset_path = "./data/SpeechCommands/"

train_set = SubsetSC(dataset_path, "training")
test_set = SubsetSC(dataset_path, "testing")
valid_set = SubsetSC(dataset_path, "validation")
print("End loading data")

print("Generate Datalaoder")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

print("Generate Datalaoder end")

# Create model
model = SC_Model_classifier(input_size=input_size,classes=classes, hidden_sizes=hidden_sizes,
                            output_sizes=output_sizes, ZeroOrderHoldRegularization=zeroOrderHoldRegularization,
                            input_bias=input_bias, bias_init=bias_init, output_bias=output_bias,
                            norm=norm, complex_output=complex_output,
                            norm_type=norm_type, B_C_init=B_C_init, stability=stability,
                            trainable_SkipLayer=trainable_skip_connections,
                            act=act,dropout=dropout)
model.to(device)

params_ssm_lr = [param for name, param in model.named_parameters() if 'B' in name or 'C' in name or 'Lambda' in name or 'log_step' in name] 
params_other_lr = [param for name, param in model.named_parameters() if 'B' not in name and 'C' not in name and 'Lambda' not in name and 'log_step' not in name]

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()

params_ssm_lr = [param for name, param in model.named_parameters() if 'B' in name or 'C' in name or 'Lambda' in name or 'log_step' in name] 
params_other_lr = [param for name, param in model.named_parameters() if 'B' not in name and 'C' not in name and 'Lambda' not in name and 'log_step' not in name]

optimizer = torch.optim.AdamW([
    {'params': params_ssm_lr, 'lr': learning_rate, 'weight_decay': 0},
    {'params': params_other_lr, 'lr': 4*learning_rate, 'weight_decay': weight_decay},
    ], lr=learning_rate, weight_decay=weight_decay)

scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=epochs*len(train_loader),
                                          cycle_mult=1.0,
                                          max_lr=[learning_rate,4*learning_rate],
                                          min_lr=[0,0],
                                          warmup_steps=100,
                                          gamma=1,
)

# Train the model
best_val_loss = 1e3  # Init
best_val_loss_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0


# subsets of the data
df_metric = pd.DataFrame(columns=['train_loss', 'train_acc', 'valid_loss','valid_acc', 'epoch', 'learning_rate', 'training_time'])
df_sub_epoch = pd.DataFrame()

if augments == 'none':
    augments = None
elif augments == 'weak':
    augments = augments_weak
elif augments == 'strong':
    augments = augments_strong
else:
    raise ValueError(f"Augments: {augments} not supported")


torch.save(model.state_dict(), os.path.join(model_save_path, 'init_model.pt'))
print("Start training")
start_time = get_time()

for epoch in range(epochs):
    train_loss, train_acc, sub_epoch_info = train_one_epoch(model, criterion, optimizer, train_loader, regularize=True, scheduler=scheduler, sub_epoch_documentation=10, augments_use=augments)
    # train_loss, train_acc, sub_epoch_info = train_one_epoch(model, criterion, optimizer, train_loader, regularize=False, scheduler=scheduler, sub_epoch_documentation=10, augments_use=augments)
    
    valid_loss, val_acc = evaluate(model, criterion, valid_loader)

    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        best_val_loss_epoch = epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_loss_model.pt'))
    if val_acc >= best_val_acc:
        best_val_acc_epoch = epoch
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_acc_model.pt'))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}_model.pt'))

    # lr_tmp = optimizer.param_groups[0]['lr']
    df_new_row = {'train_loss': train_loss,
                  'train_acc': train_acc,
                  'valid_loss': valid_loss,
                  'valid_acc': val_acc,
                  'epoch': epoch + 1,
                #   'learning_rate': lr_tmp,
                  'training_time': get_time() - start_time,
                  'learning_rate': scheduler.max_lr,
                  }
    
    
    # scheduler.step()


    df_metric.loc[epoch] = df_new_row
    df_metric.to_csv(os.path.join(model_save_path, train_history_csv))
    print(f"Epoch {epoch+1}, train_loss={train_loss:6.4f}, train_acc={train_acc:6.4f} val_loss={valid_loss:6.4f},  val_acc={val_acc:6.4f}")

    # save sub epoch info
    new_row = pd.DataFrame(sub_epoch_info)
    new_row['epoch'] =new_row['epoch']+ epoch
    df_sub_epoch = pd.concat([df_sub_epoch, new_row], ignore_index=True).reset_index(drop=True)
    df_sub_epoch.to_csv(os.path.join(model_save_path, train_sub_epoch_history_csv))


print('Evaluate on test set')
model.load_state_dict(torch.load(os.path.join(model_save_path, f'best_valid_loss_model.pt'),weights_only=True))
test_loss, test_acc = evaluate(model, criterion, test_loader)

params = model.get_number_of_parameters()
macs = model.get_number_of_MACs()

dict_test = {
    'test_loss': test_loss,
    'test_acc': test_acc,
    'best_val_loss': best_val_loss,
    'best_val_loss_epoch': best_val_loss_epoch,
    'best_val_acc': best_val_acc,
    'best_val_acc_epoch': best_val_acc_epoch,
    'params': params,
    'macs': macs,
}

# save test results to csv
df_test = pd.DataFrame(dict_test, index=[0])
df_test.to_csv(os.path.join(model_save_path, 'test_results.csv'))
