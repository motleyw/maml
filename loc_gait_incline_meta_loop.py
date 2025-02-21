import  torch
from    torch import nn, optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np
from    copy import deepcopy
import os
from    loc_gait_incline_inner_NN import LocGaitIncNN


class GaitMeta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, loc_head_config, gait_head_config, incline_head_config):
        """

        :param args:
        """
        super(GaitMeta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.weight_decay = args.weight_decay

        self.w_gait = 0.4
        self.w_incline = 0.2
        self.w_loc = 0.4

        # self.net = LocGaitIncNN(config, attention_layer_config, loc_head_config, gait_head_config, incline_head_config)
        self.net = LocGaitIncNN(config, loc_head_config, gait_head_config, incline_head_config)
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=self.weight_decay)

        # self.save_dir = 'results/loc_gait_incline_0114_run1/saved_tensors'
        # os.makedirs(self.save_dir, exist_ok=True)
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def adjust_softmax_thresholds(self, softmax_outputs, current_phases):
        """
        Adjust the softmax outputs to enforce sequential gait phase transition in a vectorized way.
        :param softmax_outputs: Tensor of softmax outputs, shape [batch_size, num_classes]
        :param current_phases: Tensor of the current predicted phase for each element in the batch, shape [batch_size]
        :return: Adjusted softmax outputs, shape [batch_size, num_classes]
        """
        batch_size, num_classes = softmax_outputs.shape

        # Define adjustment factors for each phase
        adjustment_matrix = torch.tensor([
            [1, 1, 0, 0],  # Adjustment for phase 0 (increase phase 1 and 2 likelihood)
            [0, 1, 1, 0],  # Adjustment for phase 1 (increase phase 2 and 3 likelihood)
            [0, 0, 1, 1],  # Adjustment for phase 2 (increase phase 3 and 4 likelihood)
            [1, 0, 0, 1]  # Adjustment for phase 3 (increase phase 1 and 4 likelihood)
        ], device=softmax_outputs.device, dtype=torch.float)

        # Create the adjustment factors tensor for the entire batch
        # Use advanced indexing to apply adjustment factors for each phase in the batch
        adjustment_factors = adjustment_matrix[current_phases]

        # Apply the adjustment factors to the softmax outputs
        adjusted_outputs = softmax_outputs * adjustment_factors

        # Re-normalize to ensure valid probability distribution
        adjusted_outputs = F.softmax(adjusted_outputs, dim=1)

        return adjusted_outputs

    def forward(self, x_spt, y_gait_spt, y_incline_spt, y_loc_spt, x_qry, y_gait_qry, y_incline_qry, y_loc_qry):
        """
        :param x_spt:   [batchsz, setsz, n_input, data_length]
        :param y_spt:   [batchsz, setsz]
        :param x_qry:   [batchsz, querysz, n_input, data_length]
        :param y_qry:   [batchsz, querysz]
        :return:
        """
        # print('x_spt:', x_spt.size())
        # print('y_spt:', y_spt.size())
        # print('x_qry:', x_qry.size(1))
        task_num, setsz, n_input, data_length = x_spt.size()
        # task_num, setsz, n_input= x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects_gait = [0 for _ in range(self.update_step + 1)]
        incline_mse = [0 for _ in range(self.update_step + 1)]
        corrects_loc = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # logits = self.net(x_spt[i], vars=None, bn_training=True)
            loc_output, gait_output, incline_output = self.net(x_spt[i], vars=None, bn_training=True)

            # Apply gait phase constraints to enforce sequence progression
            #adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output, dim=1), y_gait_spt[i])
            #gait_output = adjusted_gait_outputs

            # Compute individual losses
            classification_loss = F.cross_entropy(gait_output, y_gait_spt[i].long())
            incline_loss = F.mse_loss(incline_output.squeeze(), y_incline_spt[i])
            # Compute locomotion classification loss
            loc_classification_loss = F.cross_entropy(loc_output, y_loc_spt[i].long())

            # Fixed weights for support set at beginning
            weight_classification = self.w_gait
            weight_incline = self.w_incline
            weight_loc = self.w_loc

            combined_loss = (weight_classification * classification_loss +
                             weight_incline * incline_loss +
                             weight_loc * loc_classification_loss )

            # Gradient calculation and update for inner loop adaptation
            grad = torch.autograd.grad(combined_loss, self.net.parameters())

            # Clip gradients using custom function
            # self.clip_grad_by_norm_(grad, max_norm=1.0)  # max_norm is the maximum allowable norm
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                loc_output_q, gait_output_q, incline_output_q = self.net(x_qry[i], fast_weights, bn_training=True)

                # Apply gait phase constraints
                # softmax_outputs = F.softmax(gait_output_q, dim=1)
                #adjusted_gait_outputs  = self.adjust_softmax_thresholds(F.softmax(gait_output_q, dim=1), y_gait_qry[i])
                #gait_output_q = adjusted_gait_outputs

                # Save predicted incline angles
                # torch.save(incline_output_q, os.path.join(self.save_dir, f"predicted_incline_task_{i}.pt"))

                # Save true incline angles
                # torch.save(y_incline_qry[i], os.path.join(self.save_dir, f"true_incline_task_{i}.pt"))

                # Calculate query losses
                query_classification_loss = F.cross_entropy(gait_output_q, y_gait_qry[i].long())
                query_incline_loss = F.mse_loss(incline_output_q.squeeze(), y_incline_qry[i])
                query_loc_classification_loss = F.cross_entropy(loc_output_q, y_loc_qry[i].long())

                # Calculate losses for meta-update with dynamic weights
                loss_q = (self.w_gait * query_classification_loss +
                                 self.w_incline * query_incline_loss+
                          self.w_loc * query_loc_classification_loss)

                # # Accumulate losses for meta-update
                # loss_q = query_classification_loss + query_incline_loss

                losses_q[0] += loss_q

                # Calculate accuracy for gait phases
                pred_gait_q = gait_output_q.argmax(dim=1)
                correct_gait = torch.eq(pred_gait_q, y_gait_qry[i]).sum().item()
                corrects_gait[0] += correct_gait

                # Calculate accuracy for incline
                incline_mse[0] += query_incline_loss.sum().cpu()

                pred_loc_q = loc_output_q.argmax(dim=1)
                correct_loc = torch.eq(pred_loc_q, y_loc_qry[i]).sum().item()
                corrects_loc[0] += correct_loc

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                loc_output_q, gait_output_q, incline_output_q = self.net(x_qry[i], fast_weights, bn_training=True)

                # Save predicted incline angles
                # torch.save(incline_output_q, os.path.join(self.save_dir, f"predicted_incline_task_{i}.pt"))

                # Save true incline angles
                # torch.save(y_incline_qry[i], os.path.join(self.save_dir, f"true_incline_task_{i}.pt"))

                # Apply gait phase constraints
                #adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output_q, dim=1), y_gait_qry[i])
                #gait_output_q = adjusted_gait_outputs

                query_classification_loss = F.cross_entropy(gait_output_q, y_gait_qry[i].long())
                query_incline_loss = F.mse_loss(incline_output_q.squeeze(), y_incline_qry[i])
                query_loc_classification_loss = F.cross_entropy(loc_output_q, y_loc_qry[i].long())

                loss_q = (self.w_gait * query_classification_loss +
                                 self.w_incline * query_incline_loss +
                          self.w_loc * query_loc_classification_loss)
                # loss_q = query_classification_loss + query_incline_loss
                losses_q[1] += loss_q

                # Calculate accuracy for gait phases
                pred_gait_q = gait_output_q.argmax(dim=1)
                correct_gait = torch.eq(pred_gait_q, y_gait_qry[i]).sum().item()
                corrects_gait[1] += correct_gait

                incline_mse[1] += query_incline_loss.sum().cpu()

                pred_loc_q = loc_output_q.argmax(dim=1)
                correct_loc = torch.eq(pred_loc_q, y_loc_qry[i]).sum().item()
                corrects_loc[1] += correct_loc

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                loc_output, gait_output, incline_output = self.net(x_spt[i], fast_weights, bn_training=True)

                # Apply gait phase constraints to enforce sequence progression
                #adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output, dim=1), y_gait_spt[i])
                #gait_output = adjusted_gait_outputs

                classification_loss = F.cross_entropy(gait_output, y_gait_spt[i].long())
                incline_loss = F.mse_loss(incline_output.squeeze(), y_incline_spt[i])
                loc_classification_loss = F.cross_entropy(loc_output, y_loc_spt[i].long())

                combined_loss = (weight_classification * classification_loss +
                             weight_incline * incline_loss +
                                 weight_loc * loc_classification_loss)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(combined_loss, fast_weights)

                # Clip gradients using custom function
                # self.clip_grad_by_norm_(grad, max_norm=1.0)  # max_norm is the maximum allowable norm

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                loc_output_q, gait_output_q, incline_output_q = self.net(x_qry[i], fast_weights, bn_training=True)

                # Save predicted incline angles
                # torch.save(incline_output_q, os.path.join(self.save_dir, f"predicted_incline_task_{i}.pt"))

                # Save true incline angles
                # torch.save(y_incline_qry[i], os.path.join(self.save_dir, f"true_incline_task_{i}.pt"))

                # Apply gait phase constraints
                #adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output_q, dim=1), y_gait_qry[i])
                #gait_output_q = adjusted_gait_outputs

                query_classification_loss = F.cross_entropy(gait_output_q, y_gait_qry[i].long())
                query_incline_loss = F.mse_loss(incline_output_q.squeeze(), y_incline_qry[i])
                query_loc_classification_loss = F.cross_entropy(loc_output_q, y_loc_qry[i].long())

                loss_q = (self.w_gait * query_classification_loss +
                                 self.w_incline * query_incline_loss +
                          self.w_loc * query_loc_classification_loss)
                # # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = query_classification_loss + query_incline_loss
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_gait_q = gait_output_q.argmax(dim=1)
                    correct_gait = torch.eq(pred_gait_q, y_gait_qry[i]).sum().item()
                    corrects_gait[k + 1] += correct_gait

                    incline_mse[k + 1] += query_incline_loss.sum().cpu()

                    pred_loc_q = loc_output_q.argmax(dim=1)
                    correct_loc = torch.eq(pred_loc_q, y_loc_qry[i]).sum().item()
                    corrects_loc[k + 1] += correct_loc

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs_gait = np.array(corrects_gait) / (querysz * task_num)
        rmse_incline = np.sqrt(np.sum(incline_mse) / (querysz * task_num))
        accs_loc = np.array(corrects_loc) / (querysz * task_num)

        return accs_gait, rmse_incline, accs_loc


    def finetunning(self, x_spt, y_gait_spt, y_incline_spt, y_loc_spt, x_qry, y_gait_qry, y_incline_qry, y_loc_qry):
        """

        :param x_spt:   [setsz, n_input, data_length]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, n_input, data_length]
        :param y_qry:   [querysz]
        :return:
        """
        # assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        corrects_gait = [0 for _ in range(self.update_step_test + 1)]
        # incline_err = [0 for _ in range(self.update_step_test + 1)]
        incline_mse = [0 for _ in range(self.update_step_test + 1)]
        corrects_loc = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        loc_output, gait_output, incline_output = net(x_spt)

        # Apply gait phase constraints to enforce sequence progression
        adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output, dim=1), y_gait_spt)
        gait_output = adjusted_gait_outputs

        # Calculate initial loss for gait, incline
        classification_loss = F.cross_entropy(gait_output, y_gait_spt.long())
        incline_loss = F.mse_loss(incline_output.squeeze(), y_incline_spt)
        loc_classification_loss = F.cross_entropy(loc_output, y_loc_spt.long())

        # Fixed weights at initial
        weight_classification = self.w_gait
        weight_incline = self.w_incline
        weight_loc = self.w_loc

        # Calculate combined loss with dynamic weights
        combined_loss = (weight_classification * classification_loss +
                         weight_incline * incline_loss +
                         weight_loc * loc_classification_loss)
        # combined_loss = classification_loss + incline_loss

        # Calculate gradients and update weights
        grad = torch.autograd.grad(combined_loss, net.parameters())
        # Clip gradients using custom function
        # self.clip_grad_by_norm_(grad, max_norm=1.0)  # max_norm is the maximum allowable norm
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            loc_output_q, gait_output_q, incline_output_q = net(x_qry, net.parameters(), bn_training=True)

            query_incline_loss = F.mse_loss(incline_output_q.squeeze(), y_incline_qry)

            # Apply gait phase constraints
            adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output_q, dim=1), y_gait_qry)

            # Calculate accuracy for gait phase
            pred_gait_q = adjusted_gait_outputs.argmax(dim=1)
            corrects_gait[0] = torch.eq(pred_gait_q, y_gait_qry).sum().item()

            # Calculate accuracy for incline
            # pred_incline_q = incline_output_q.squeeze().round()
            # incline_err[0] = torch.eq(pred_incline_q, y_incline_qry).sum().item()
            incline_mse[0] += query_incline_loss.sum().cpu()

            # Calculate accuracy for locomotion
            pred_loc_q = loc_output_q.argmax(dim=1)
            corrects_loc[0] = torch.eq(pred_loc_q, y_loc_qry).sum().item()

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            loc_output_q, gait_output_q, incline_output_q = net(x_qry, fast_weights, bn_training=True)

            query_incline_loss = F.mse_loss(incline_output_q.squeeze(), y_incline_qry)

            # Apply gait phase constraints
            adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output_q, dim=1), y_gait_qry)

            # Calculate accuracy for gait phase
            pred_gait_q = adjusted_gait_outputs.argmax(dim=1)
            corrects_gait[1] = torch.eq(pred_gait_q, y_gait_qry).sum().item()

            # Calculate accuracy for incline
            # pred_incline_q = incline_output_q.squeeze().round()
            # incline_err[1] = torch.eq(pred_incline_q, y_incline_qry).sum().item()
            incline_mse[1] += query_incline_loss.sum().cpu()

            # Calculate accuracy for locomotion
            pred_loc_q = loc_output_q.argmax(dim=1)
            corrects_loc[1] = torch.eq(pred_loc_q, y_loc_qry).sum().item()

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            loc_output, gait_output, incline_output = net(x_spt, fast_weights, bn_training=True)

            # Apply gait phase constraints to enforce sequence progression
            adjusted_gait_outputs = self.adjust_softmax_thresholds(F.softmax(gait_output, dim=1), y_gait_spt)
            gait_output = adjusted_gait_outputs

            # Calculate loss for gait, incline
            classification_loss = F.cross_entropy(gait_output, y_gait_spt.long())
            incline_loss = F.mse_loss(incline_output.squeeze(), y_incline_spt)
            loc_classification_loss = F.cross_entropy(loc_output, y_loc_spt.long())

            # Calculate combined loss with dynamic weights
            combined_loss = (self.w_gait * classification_loss +
                             self.w_incline * incline_loss +
                             self.w_loc * loc_classification_loss)
            # combined_loss = classification_loss + incline_loss

            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(combined_loss, fast_weights)

            # Clip gradients using custom function
            # self.clip_grad_by_norm_(grad, max_norm=1.0)  # max_norm is the maximum allowable norm
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            loc_output_q, gait_output_q, incline_output_q = net(x_qry, fast_weights, bn_training=True)

            # Apply gait phase constraints
            adjusted_gait_outputs = self.adjust_softmax_thresholds(gait_output_q, y_gait_qry)

            with torch.no_grad():
                pred_gait_q = adjusted_gait_outputs.argmax(dim=1)
                corrects_gait[k + 1] = torch.eq(pred_gait_q, y_gait_qry).sum().item()

                # Calculate accuracy for incline
                # pred_incline_q = incline_output_q.squeeze().round()
                # incline_err[k + 1] = torch.eq(pred_incline_q, y_incline_qry).sum().item()

                incline_mse[k + 1] += incline_loss.sum().cpu()

                pred_loc_q = loc_output_q.argmax(dim=1)
                corrects_loc[k + 1] = torch.eq(pred_loc_q, y_loc_qry).sum().item()
        del net

        accs_gait = np.array(corrects_gait) / querysz
        # errs_incline = np.array(incline_err) / querysz
        rmse_incline = np.sqrt(np.sum(incline_mse) / querysz)
        accs_loc = np.array(corrects_loc) / querysz

        return accs_gait, rmse_incline, accs_loc


def main():
    pass


if __name__ == '__main__':
    main()
