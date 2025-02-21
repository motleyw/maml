import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml

import numpy as np
# Define CNN Model
from train_utils import *
from datetime import datetime
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
#from models import *
from loc_gait_incline_meta_loop import GaitMeta
import argparse

def test_model():


    return

def find_checkpoint(folderpath: str, test_id: str, epoch: int = None) -> str:
    """
    Find and return a single .pt checkpoint filepath in `folderpath` based on `test_id`.
    Returns the first matching file found.
    Raises:
        FileNotFoundError if no matching file is found.
    """
    trained_model_dir = os.path.join(folderpath, "trained_model")
    if epoch is not None:
        # Look for files that end with _epoch{epoch}.pt
        pattern = f"*testid_{test_id}*_epoch{epoch}.pt"
    else:
        # Look for files that contain test_id but no epoch restriction
        pattern = f"*testid_{test_id}*.pt"
    # Construct full search pattern
    search_pattern = os.path.join(trained_model_dir, pattern)
    matches = glob.glob(search_pattern)
    
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for test_id={test_id}, epoch={epoch} in {trained_model_dir}.")
    return matches[0]


def load_baseline_model(folderpath,test_id,device):
    epoch=151
    checkpoint_path=find_checkpoint(folderpath=folderpath,test_id=test_id,epoch=epoch)

    with open(os.path.join(folderpath,'config.yml'), 'r') as f:
        config_dict = yaml.safe_load(f)
        #print(config_dict)
    train_args = argparse.Namespace(**config_dict["args"] )
    model = CNNModel(train_args).to(device)
    model.update_architecture(config_dict['model_config'])
    model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
    model.eval()
    print(f'load baseline model at {folderpath}, test id: {test_id}')
    return model.to(device),train_args

def load_maml_model(folderpath,test_id,device):
    with open(os.path.join(folderpath,'config.yml'), 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(config_dict)
    train_args = argparse.Namespace(**config_dict["args"] )
    #config, loc_head_config, gait_head_config, incline_head_config = get_model_config(winlen=100)
    config=config_dict['model_config']['config']
    loc_head_config=config_dict['model_config']['loc_head_config']
    gait_head_config=config_dict['model_config']['gait_head_config']
    incline_head_config=config_dict['model_config']['incline_head_config']
    model = GaitMeta(train_args, config, loc_head_config, gait_head_config, incline_head_config).to(device) 
    checkpoint_path=find_checkpoint(folderpath=folderpath,test_id=test_id)


    model.load_state_dict(torch.load(checkpoint_path,weights_only=False,map_location=torch.device('cpu'))) # Updated for CPU only device
    model.eval()
    print(f'load maml model at {folderpath}, test id: {test_id}')
    return model.to(device)

def get_loss(gait_output,incline_output,loco_output,y_gait_gt,y_incline_gt,y_loco_gt,args):
    weight_g_classification = args.w_gait
    weight_loco_classification = args.w_loco
    weight_incline = 1 - args.w_gait - args.w_loco
    gait_classification_loss = F.cross_entropy(gait_output, y_gait_gt.long())
    loco_classification_loss = F.cross_entropy(loco_output, y_loco_gt.long())
    #print(incline_output.shape)
    incline_loss = F.mse_loss(incline_output.squeeze(), y_incline_gt)
    combined_loss = (weight_g_classification * gait_classification_loss +
                    weight_loco_classification * loco_classification_loss +
                    weight_incline * incline_loss)
    return combined_loss       
 
def test_finetune(test_args,test_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    baseline_folderpath='results/baselineCNN_gait_incline_0210_run1'
    maml_folderpath='results/maml_lgi_0210'


    data_root= '../data_label'
    #train_ids = [x for x in id_list if x != test_id]
    # Get CSV file paths
    test_csv_list = get_csv_files(data_root, [test_id])

    import os
    import numpy as np
    import torch.nn.functional as F

    # Dictionary to hold all results, keyed by CSV file name
    results_dict = {}

    for csv_i in range(len(test_csv_list)):
        # prepare models
        model_baseline, train_args = load_baseline_model(baseline_folderpath, test_id, device)
        model_maml = load_maml_model(maml_folderpath, test_id, device)

        # Freeze convolution parts of baseline
        for param in model_baseline.conv1.parameters():
            param.requires_grad = False
        for param in model_baseline.bn1.parameters():
            param.requires_grad = False
        for param in model_baseline.linear1.parameters():
            param.requires_grad = False

        base_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_baseline.parameters()), 
            #lr=1e-3#
            lr=test_args.transfer_lr

        )

        # get test data, one csv file per run, spt and qry from the same session
        test_data, test_g_label, test_i_gt, test_loco_label = load_data_file_lgi(
            [test_csv_list[csv_i]], 
            winlen=train_args.window_length, 
            max_samples=test_args.max_samples
        )
        test_dataloader = prepare_dataloader(
            test_data, test_g_label, test_i_gt, test_loco_label, 
            device, batch_size=test_args.batchsize
        )

        # Check that there's enough data for support + query
        needed_samples = test_args.spt_batch * test_args.batchsize * 2
        if len(test_dataloader.dataset) < needed_samples:
            print(f"Not enough data, skipping test of {test_csv_list[csv_i]}")
            continue

        # Collect support batches
        dataloader_iter = iter(test_dataloader)
        spt_batches = []
        for i in range(test_args.spt_batch):
            data, gait_gt, incline_gt, loco_gt = next(dataloader_iter)
            spt_batches.append((data, gait_gt, incline_gt, loco_gt))

        # Collect all remaining as query batches
        qry_batches = []
        while True:
            try:
                data, gait_gt, incline_gt, loco_gt = next(dataloader_iter)
                qry_batches.append((data, gait_gt, incline_gt, loco_gt))
            except StopIteration:
                break

        # Initialize "fast weights" for the MAML net
        fast_weights = list(model_maml.net.parameters())  # or model_maml.net.parameters()

        # Prepare lists to store performance for each update (0..4)
        maml_g_acc_list = []
        maml_l_acc_list = []
        maml_i_rmse_list = []

        base_g_acc_list = []
        base_l_acc_list = []
        base_i_rmse_list = []

        # 5 gradient update steps on the support set
        for update in range(10):


            # 4) Evaluate on query set
            #    We'll compute gait accuracy, loco accuracy, and incline MSE for both MAML & baseline
            te_gait_correct_m = 0
            te_loco_correct_m = 0
            te_incline_sum_m = 0
            te_total_samples_m = 0

            te_gait_correct_b = 0
            te_loco_correct_b = 0
            te_incline_sum_b = 0
            te_total_samples_b = 0

            with torch.no_grad():
                for (data, gait_gt, incline_gt, loco_gt) in qry_batches:
                    data = data.to(device)
                    gait_gt = gait_gt.to(device)
                    incline_gt = incline_gt.to(device)
                    loco_gt = loco_gt.to(device)

                    # MAML forward
                    l_m, g_m, i_m = model_maml.net(data, fast_weights, bn_training=True)
                    # Gait accuracy
                    pred_g_m = g_m.argmax(dim=1)
                    te_gait_correct_m += (pred_g_m == gait_gt).sum().item()

                    # Loco accuracy
                    pred_l_m = l_m.argmax(dim=1)
                    te_loco_correct_m += (pred_l_m == loco_gt).sum().item()

                    # Incline MSE (accumulate sum of squared errors, will divide later)
                    # i_m shape: [batch_size, 1], so let's flatten it
                    mse_m = F.mse_loss(i_m.squeeze(), incline_gt, reduction='sum')
                    te_incline_sum_m += mse_m.item()
                    te_total_samples_m += gait_gt.size(0)

                    # Baseline forward
                    g_b, i_b, l_b = model_baseline(data)
                    # Gait accuracy
                    pred_g_b = g_b.argmax(dim=1)
                    te_gait_correct_b += (pred_g_b == gait_gt).sum().item()

                    # Loco accuracy
                    pred_l_b = l_b.argmax(dim=1)
                    te_loco_correct_b += (pred_l_b == loco_gt).sum().item()

                    # Incline MSE
                    mse_b = F.mse_loss(i_b.squeeze(), incline_gt, reduction='sum')
                    te_incline_sum_b += mse_b.item()
                    te_total_samples_b += gait_gt.size(0)

            # Compute final metrics
            # MAML
            g_acc_maml = te_gait_correct_m / te_total_samples_m
            l_acc_maml = te_loco_correct_m / te_total_samples_m
            i_rmse_maml = np.sqrt(te_incline_sum_m / te_total_samples_m)

            # Baseline
            g_acc_base = te_gait_correct_b / te_total_samples_b
            l_acc_base = te_loco_correct_b / te_total_samples_b
            i_rmse_base = np.sqrt(te_incline_sum_b / te_total_samples_b)

            # Store them
            maml_g_acc_list.append(g_acc_maml)
            maml_l_acc_list.append(l_acc_maml)
            maml_i_rmse_list.append(i_rmse_maml)

            base_g_acc_list.append(g_acc_base)
            base_l_acc_list.append(l_acc_base)
            base_i_rmse_list.append(i_rmse_base)

            
            maml_loss_spt = 0.0
            base_loss_spt = 0.0

            # 1) Accumulate loss over support batches
            for (data, gait_gt, incline_gt, loco_gt) in spt_batches:
                data = data.to(device)
                gait_gt = gait_gt.to(device)
                incline_gt = incline_gt.to(device)
                loco_gt = loco_gt.to(device)

                # MAML forward
                l_m, g_m, i_m = model_maml.net(data, fast_weights, bn_training=True)
                loss_m = get_loss(g_m, i_m, l_m, gait_gt, incline_gt, loco_gt, test_args)
                maml_loss_spt += loss_m

                # Baseline forward
                g_b, i_b, l_b = model_baseline(data)
                loss_b = get_loss(g_b, i_b, l_b, gait_gt, incline_gt, loco_gt, test_args)
                base_loss_spt += loss_b

            # 2) Fine-tune MAML
            grad_m = torch.autograd.grad(maml_loss_spt, fast_weights, create_graph=False)
            updated_fast_weights = []
            for (g, w) in zip(grad_m, fast_weights):
                updated_fast_weights.append(w - test_args.maml_lr * g)
            fast_weights = updated_fast_weights

            # 3) Fine-tune Baseline
            base_optimizer.zero_grad()
            base_loss_spt.backward()
            base_optimizer.step()
        # After finishing all updates for this csv file, store results in the dictionary
        csv_file_name = os.path.basename(test_csv_list[csv_i])
        results_dict[csv_file_name] = {
            'maml_g_acc': maml_g_acc_list,
            'maml_l_acc': maml_l_acc_list,
            'maml_i_rmse': maml_i_rmse_list,
            'base_g_acc': base_g_acc_list,
            'base_l_acc': base_l_acc_list,
            'base_i_rmse': base_i_rmse_list
        }

    # Finally, save the dictionary as an .npz file in "result/finetune test/" named after test_id
    
    save_path = os.path.join(test_args.savepath, f"{test_id}.npz")
    np.savez(save_path,  results_dict=results_dict, allow_pickle=True)
    print(f"\nSaved fine-tune metrics to {save_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batchsize', type=int, default=200)
    argparser.add_argument('--runs', type=int, default=5)
    argparser.add_argument('--spt_batch', type=int, default=1)
    argparser.add_argument('--maml_lr', type=float, default=2e-5)
    argparser.add_argument('--transfer_lr', type=float, default=1e-3)#1e-3)
    argparser.add_argument('--w-gait', type=float, help='weight for gait classification loss', default=0.4)
    argparser.add_argument('--w-loco', type=float, help='weight for loco classification loss', default=0.2)
    argparser.add_argument('--max_samples', type=int, help='frames to be included per session from start', default=6000)
    argparser.add_argument('--savepath', type=str, help='path to save', default='testresult/0211_run3')
    test_args = argparser.parse_args()
    args_dict = vars(test_args)
    save_config = {
    'args': args_dict,
    }
    os.makedirs(test_args.savepath, exist_ok=True)
    with open(os.path.join(test_args.savepath, 'testconfig.yml'), 'w') as yaml_file:
        yaml.dump(save_config, yaml_file, default_flow_style=False)

    #test_id='Sub07'
    id_list = ['Sub01', 'Sub02', 'Sub03', 'Sub04', 'Sub05', 'Sub06', 'Sub07', 'Sub08', 'Sub09']
    for test_id in id_list:
        test_finetune(test_args,test_id)

'''
    for csv_i in range(len(test_csv_list)):
        # prepare models
        model_baseline,train_args = load_baseline_model(baseline_folderpath,test_id,device)
        model_maml = load_maml_model(maml_folderpath,test_id,device)
        #froze conv layer
        for param in model_baseline.conv1.parameters():
            param.requires_grad = False
        for param in model_baseline.bn1.parameters():
            param.requires_grad = False
        for param in model_baseline.linear1.parameters():
            param.requires_grad = False
        base_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_baseline.parameters()), 
            lr=test_args.finetune_lr
        )
        # get test data, one csv file per run, spt and qry are from the same session
        test_data, test_g_label, test_i_gt, test_loco_label = load_data_file_lgi([test_csv_list[csv_i]], winlen=train_args.window_length, max_samples=test_args.max_samples)
        test_dataloader = prepare_dataloader(test_data, test_g_label, test_i_gt, test_loco_label, device, batch_size=test_args.batchsize)
        
        if len(test_dataloader.dataset) < (test_args.spt_batch * test_args.batchsize *2):
            print(f"Not enough data, skipping test of {test_csv_list[csv_i]}")
            continue
        #spt and qry 
        dataloader_iter=iter(test_dataloader)
        spt_batches=[]
        for i in range(test_args.spt_batch):
            data, gait_gt, incline_gt, loco_gt = next(dataloader_iter)
            spt_batches.append((data, gait_gt, incline_gt, loco_gt))
        qry_batches=[] 
        while True:
            try:
                data, gait_gt, incline_gt, loco_gt = next(dataloader_iter)
                qry_batches.append((data, gait_gt, incline_gt, loco_gt))
            except StopIteration:
                break

        fast_weights=model_maml.net.parameters()
        for update in range(5):
            maml_loss_spt=0
            base_loss_spt=0
            for (data, gait_gt, incline_gt, loco_gt) in spt_batches:
                data, gait_gt, incline_gt, loco_gt = data.to(device), gait_gt.to(device), incline_gt.to(device), loco_gt.to(device)
                g_m, l_m, i_m=model_maml.net(data,fast_weights,bn_training=True) # !! wrong order somehow

                maml_loss=get_loss(g_m, i_m, l_m, gait_gt, incline_gt, loco_gt,test_args)
                g_b, i_b, l_b=model_baseline(data)
                base_loss=get_loss(g_b, i_b, l_b, gait_gt, incline_gt, loco_gt,test_args)
                maml_loss_spt+=maml_loss
                base_loss_spt+=base_loss
            #finetune maml
            grad_m=torch.autograd.grad(maml_loss_spt,model_maml.net.parameters())
            fast_weights=list(map(lambda p:p[1]-test_args.finetune_lr*p[0],zip(grad_m,model_maml.net.parameters())))
            #finetune base 
            base_optimizer.zero_grad()
            base_loss_spt.backward()
            base_optimizer.step()


            te_gait_correct, te_loco_correct, te_total_samples, te_total_incline_diff = 0, 0, 0, 0
            with torch.no_grad():
                for (data, gait_gt, incline_gt, loco_gt) in qry_batches:
                    data, gait_gt, incline_gt, loco_gt = data.to(device), gait_gt.to(device), incline_gt.to(device), loco_gt.to(device)
                    g_m, l_m, i_m=model_maml.net(data,fast_weights,bn_training=True) # !! wrong order somehow
                    maml_loss=get_loss(g_m, i_m, l_m, gait_gt, incline_gt, loco_gt,test_args)
                    g_b, i_b, l_b=model_baseline(data)
                    base_loss=get_loss(g_b, i_b, l_b, gait_gt, incline_gt, loco_gt,test_args)
'''




