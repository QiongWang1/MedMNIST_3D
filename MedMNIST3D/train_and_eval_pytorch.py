import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from tqdm import trange
from utils import Transform3D, model_to_syncbn
from mito_dataset import MitoPatchDataset
from sklearn.metrics import roc_auc_score, accuracy_score


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run):

    lr = 0.00001
    gamma = 0.5 
    milestones = [0.6 * num_epochs, 0.8 * num_epochs]

    # Early stopping parameters
    patience = 10 
    best_val_loss = float('inf')
    patience_counter = 0
    
    # L2 regularization parameter
    weight_decay = 1e-3



    # info = INFO[data_flag]
    # task = info['task']
    # n_channels = 3 if as_rgb else info['n_channels']
    # n_classes = len(info['label'])
    n_channels = 1     # grey image (1, H, W)
    n_classes = 2      # binary: mito / not mito

    # DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 

    
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
    

    # train_dataset = MitoPatchDataset(
    #     csv_path='mito_labels.csv',
    #     data_dir='mito_downloaded_patch/',
    #     transform=train_transform
    # )
    # train_dataset_at_eval = MitoPatchDataset(
    #     csv_path='mito_labels.csv',
    #     data_dir='mito_downloaded_patch/',
    #     transform=eval_transform
    # )
    # val_dataset = train_dataset_at_eval  
    # test_dataset = train_dataset_at_eval 


    train_dataset = MitoPatchDataset(
        csv_path='mito_train.csv',
        data_dir='mito_downloaded_patch/',
        transform=train_transform
    )
    val_dataset = MitoPatchDataset(
        csv_path='mito_val.csv',
        data_dir='mito_downloaded_patch/',
        transform=eval_transform
    )
    test_dataset = MitoPatchDataset(
        csv_path='mito_test.csv',
        data_dir='mito_downloaded_patch/',
        transform=eval_transform
    )



    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    train_loader_at_eval = data.DataLoader(dataset=train_dataset, 
                                       batch_size=batch_size,
                                       shuffle=False)

    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')

    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))

    model = model.to(device)

    # train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    # val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    # test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)
    train_evaluator = None
    val_evaluator = None
    test_evaluator = None

    criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)

        train_metrics = test(model, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(model, val_loader, criterion, device, run, output_root)
        test_metrics = test(model, test_loader, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    # Modified optimizer with L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    # Training loop with early stopping
    for epoch in trange(num_epochs):
    
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)
    
        train_metrics = test(model, train_loader_at_eval, criterion, device, run, None)
        val_metrics = test(model, val_loader, criterion, device, run, None)
        test_metrics = test(model, test_loader, criterion, device, run, None)
        
        val_loss = val_metrics[0]  # Get validation loss for early stopping
                    
        scheduler.step()
    
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
        
        # Early stopping logic based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'Epoch {epoch}: Validation loss improved to {val_loss:.5f}')
        else:
            patience_counter += 1
            print(f'Epoch {epoch}: Val Loss = {val_loss:.5f}, Best Val Loss = {best_val_loss:.5f}, Patience Counter = {patience_counter}/{patience}')
        
        # Check if we should stop early
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.5f}')
            break
        
        # Best model selection based on AUC (keeping original logic)
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)

            print('Current best AUC:', best_auc)
            print('Current best epoch:', best_epoch)

    state = {
        'net': best_model.state_dict(),  # Save best model instead of final model
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    # Final evaluation with best model
    train_metrics = test(best_model, train_loader_at_eval, criterion, device, run, output_root)
    val_metrics = test(best_model, val_loader, criterion, device, run, output_root)
    test_metrics = test(best_model, test_loader, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
    print(log)

    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)        
        
    writer.close()


def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss



def test(model, data_loader, criterion, device, run, save_folder=None):

    model.eval()
    total_loss = []
    targets_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

            probs = nn.Softmax(dim=1)(outputs)
            total_loss.append(loss.item())

            targets_all.extend(targets.cpu().numpy())
            y_pred_all.extend(probs[:, 1].detach().cpu().numpy())  

    # Compute metrics
    auc = roc_auc_score(targets_all, y_pred_all)
    pred_labels = (np.array(y_pred_all) > 0.5).astype(int)
    acc = accuracy_score(targets_all, pred_labels)

    test_loss = sum(total_loss) / len(total_loss)

    # Optional debug prints (can be commented out for cleaner output)
    print("Ground Truth:", targets_all)
    print("Predicted Probabilities (class 1):", y_pred_all)
    print("Predicted Labels:", pred_labels)

    return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--conv',
                        default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    run = args.run

    # Run training
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run)











# import argparse
# import os
# import time
# from collections import OrderedDict
# from copy import deepcopy

# import medmnist
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
# from medmnist import INFO, Evaluator
# from models import ResNet18, ResNet50
# from tensorboardX import SummaryWriter
# from tqdm import trange
# from utils import Transform3D, model_to_syncbn
# from mito_dataset import MitoPatchDataset
# from sklearn.metrics import roc_auc_score, accuracy_score


# def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run):

#     lr = 0.0001 
#     gamma = 0.1
#     milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    

#     patience = 5
#     best_val_loss = float('inf')
#     patience_counter = 0



#     # info = INFO[data_flag]
#     # task = info['task']
#     # n_channels = 3 if as_rgb else info['n_channels']
#     # n_classes = len(info['label'])
#     n_channels = 1     # grey image (1, H, W)
#     n_classes = 2      # binary：mito / not mito

#     # DataClass = getattr(medmnist, info['python_class'])

#     str_ids = gpu_ids.split(',')
#     gpu_ids = []
#     for str_id in str_ids:
#         id = int(str_id)
#         if id >= 0:
#             gpu_ids.append(id)
#     if len(gpu_ids) > 0:
#         os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

#     device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 

    
#     output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
#     if not os.path.exists(output_root):
#         os.makedirs(output_root)

#     print('==> Preparing data...')

#     train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
#     eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
    

#     # train_dataset = MitoPatchDataset(
#     #     csv_path='mito_labels.csv',
#     #     data_dir='mito_downloaded_patch/',
#     #     transform=train_transform
#     # )
#     # train_dataset_at_eval = MitoPatchDataset(
#     #     csv_path='mito_labels.csv',
#     #     data_dir='mito_downloaded_patch/',
#     #     transform=eval_transform
#     # )
#     # val_dataset = train_dataset_at_eval  
#     # test_dataset = train_dataset_at_eval 


#     train_dataset = MitoPatchDataset(
#         csv_path='mito_train.csv',
#         data_dir='mito_downloaded_patch/',
#         transform=train_transform
#     )
#     val_dataset = MitoPatchDataset(
#         csv_path='mito_val.csv',
#         data_dir='mito_downloaded_patch/',
#         transform=eval_transform
#     )
#     test_dataset = MitoPatchDataset(
#         csv_path='mito_test.csv',
#         data_dir='mito_downloaded_patch/',
#         transform=eval_transform
#     )



#     train_loader = data.DataLoader(dataset=train_dataset,
#                                 batch_size=batch_size,
#                                 shuffle=True)

#     train_loader_at_eval = data.DataLoader(dataset=train_dataset, 
#                                        batch_size=batch_size,
#                                        shuffle=False)

#     val_loader = data.DataLoader(dataset=val_dataset,
#                                 batch_size=batch_size,
#                                 shuffle=False)
#     test_loader = data.DataLoader(dataset=test_dataset,
#                                 batch_size=batch_size,
#                                 shuffle=False)

#     print('==> Building and training model...')

#     if model_flag == 'resnet18':
#         model = ResNet18(in_channels=n_channels, num_classes=n_classes)
#     elif model_flag == 'resnet50':
#         model = ResNet50(in_channels=n_channels, num_classes=n_classes)
#     else:
#         raise NotImplementedError

#     if conv=='ACSConv':
#         model = model_to_syncbn(ACSConverter(model))
#     if conv=='Conv2_5d':
#         model = model_to_syncbn(Conv2_5dConverter(model))
#     if conv=='Conv3d':
#         if pretrained_3d == 'i3d':
#             model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
#         else:
#             model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))

#     model = model.to(device)

#     # train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
#     # val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
#     # test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)
#     train_evaluator = None
#     val_evaluator = None
#     test_evaluator = None

#     criterion = nn.CrossEntropyLoss()

#     if model_path is not None:
#         model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)

#         train_metrics = test(model, train_loader_at_eval, criterion, device, run, output_root)
#         val_metrics = test(model, val_loader, criterion, device, run, output_root)
#         test_metrics = test(model, test_loader, criterion, device, run, output_root)

#         print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
#               'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
#               'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

#     if num_epochs == 0:
#         return


#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

#     logs = ['loss', 'auc', 'acc']
#     train_logs = ['train_'+log for log in logs]
#     val_logs = ['val_'+log for log in logs]
#     test_logs = ['test_'+log for log in logs]
#     log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

#     writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

#     best_auc = 0
#     best_epoch = 0
#     best_model = deepcopy(model)

#     global iteration
#     iteration = 0

#     for epoch in trange(num_epochs):
    
#         train_loss = train(model, train_loader, criterion, optimizer, device, writer)
    
#         train_metrics = test(model, train_loader_at_eval, criterion, device, run, None)
#         val_metrics = test(model, val_loader, criterion, device, run, None)
#         test_metrics = test(model, test_loader, criterion, device, run, None)
                    
#         scheduler.step()
    
#         for i, key in enumerate(train_logs):
#             log_dict[key] = train_metrics[i]
#         for i, key in enumerate(val_logs):
#             log_dict[key] = val_metrics[i]
#         for i, key in enumerate(test_logs):
#             log_dict[key] = test_metrics[i]

#         for key, value in log_dict.items():
#             writer.add_scalar(key, value, epoch)
        
#         cur_auc = val_metrics[1]
#         if cur_auc > best_auc:
#             best_epoch = epoch
#             best_auc = cur_auc
#             best_model = deepcopy(model)

#             print('cur_best_auc:', best_auc)
#             print('cur_best_epoch', best_epoch)

#     state = {
#         'net': model.state_dict(),
#     }

#     path = os.path.join(output_root, 'best_model.pth')
#     torch.save(state, path)


#     train_metrics = test(best_model, train_loader_at_eval, criterion, device, run, output_root)
#     val_metrics = test(best_model, val_loader, criterion, device, run, output_root)
#     test_metrics = test(best_model, test_loader, criterion, device, run, output_root)

#     train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
#     val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
#     test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

#     log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
#     print(log)

#     with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
#         f.write(log)        
        
#     writer.close()


# def train(model, train_loader, criterion, optimizer, device, writer):
#     total_loss = []
#     global iteration

#     model.train()
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs.to(device))

#         targets = torch.squeeze(targets, 1).long().to(device)
#         loss = criterion(outputs, targets)

#         total_loss.append(loss.item())
#         writer.add_scalar('train_loss_logs', loss.item(), iteration)
#         iteration += 1

#         loss.backward()
#         optimizer.step()

#     epoch_loss = sum(total_loss)/len(total_loss)
#     return epoch_loss



# def test(model, data_loader, criterion, device, run, save_folder=None):

#     model.eval()
#     total_loss = []
#     targets_all = []
#     y_pred_all = []

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(data_loader):
#             outputs = model(inputs.to(device))

#             targets = torch.squeeze(targets, 1).long().to(device)
#             loss = criterion(outputs, targets)

#             probs = nn.Softmax(dim=1)(outputs)
#             total_loss.append(loss.item())

#             targets_all.extend(targets.cpu().numpy())
#             y_pred_all.extend(probs[:, 1].detach().cpu().numpy())  

#     # compute metrics
#     auc = roc_auc_score(targets_all, y_pred_all)
#     pred_labels = (np.array(y_pred_all) > 0.5).astype(int)
#     acc = accuracy_score(targets_all, pred_labels)

#     test_loss = sum(total_loss) / len(total_loss)
#     pred_labels = (np.array(y_pred_all) > 0.5).astype(int)

#     print("📌 Ground Truth:", targets_all)
#     print("📌 Predicted Probabilities (class 1):", y_pred_all)
#     print("📌 Predicted Labels:", pred_labels)

#     return [test_loss, auc, acc]

#     # test_loss = sum(total_loss) / len(total_loss)
#     # return [test_loss, auc, acc]
#     # print("Ground Truth:", targets_all)
#     # print("Predicted Probs:", y_pred_all)





# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='RUN Baseline model of MedMNIST3D')

#     parser.add_argument('--data_flag',
#                         default='organmnist3d',
#                         type=str)
#     parser.add_argument('--output_root',
#                         default='./output',
#                         help='output root, where to save models',
#                         type=str)
#     parser.add_argument('--num_epochs',
#                         default=100,
#                         help='num of epochs of training, the script would only test model if set num_epochs to 0',
#                         type=int)
#     parser.add_argument('--size',
#                         default=28,
#                         help='the image size of the dataset, 28 or 64, default=28',
#                         type=int)
#     parser.add_argument('--gpu_ids',
#                         default='0',
#                         type=str)
#     parser.add_argument('--batch_size',
#                         default=32,
#                         type=int)
#     parser.add_argument('--conv',
#                         default='ACSConv',
#                         help='choose converter from Conv2_5d, Conv3d, ACSConv',
#                         type=str)
#     parser.add_argument('--pretrained_3d',
#                         default='i3d',
#                         type=str)
#     parser.add_argument('--download',
#                         action="store_true")
#     parser.add_argument('--as_rgb',
#                         help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
#                         action="store_true")
#     parser.add_argument('--shape_transform',
#                         help='for shape dataset, whether multiply 0.5 at eval',
#                         action="store_true")
#     parser.add_argument('--model_path',
#                         default=None,
#                         help='root of the pretrained model to test',
#                         type=str)
#     parser.add_argument('--model_flag',
#                         default='resnet18',
#                         help='choose backbone, resnet18/resnet50',
#                         type=str)
#     parser.add_argument('--run',
#                         default='model1',
#                         help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
#                         type=str)


#     args = parser.parse_args()
#     data_flag = args.data_flag
#     output_root = args.output_root
#     num_epochs = args.num_epochs
#     size = args.size
#     gpu_ids = args.gpu_ids
#     batch_size = args.batch_size
#     conv = args.conv
#     pretrained_3d = args.pretrained_3d
#     download = args.download
#     model_flag = args.model_flag
#     as_rgb = args.as_rgb
#     model_path = args.model_path
#     shape_transform = args.shape_transform
#     run = args.run

# # test
#     main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run)








