import  torch
import  numpy as np
from    torch import nn
from    torch.nn import functional as F


class LocGaitIncNN(nn.Module):
    """

    """

    def __init__(self, config, loc_head_config, gait_head_config, incline_head_config):
        """

        :param config: network config file, type:list of (string, list)

        """
        super(LocGaitIncNN, self).__init__()

        self.config = config
        # self.attention_layer_config = attention_layer_config
        self.loc_head_config = loc_head_config
        self.gait_head_config = gait_head_config
        self.incline_head_config = incline_head_config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        # Build the layers from config
        for i, (name, param) in enumerate(self.config):
            if name == 'conv1d':
                # [out_channels, in_channels, kernel_size]
                w = nn.Parameter(torch.ones(*param[:3]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # Bias for Conv1D: [out_channels]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'bn':
                # [ch_out] - weights and biases for batch normalization
                weight = nn.Parameter(torch.ones(param[0]))
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(weight)
                self.vars.append(bias)

                # must set requires_grad=False for running mean and variance
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['relu', 'flatten', 'dropout']:
                continue
            else:
                raise NotImplementedError

        # Append reduced layer parameters
        # for name, param in self.attention_layer_config:
        #     if name == 'attention':
        #         attention_w = nn.Parameter(torch.ones(param[1], param[0]))  # Shape: [attention_size, input_size]
        #         attention_b = nn.Parameter(torch.zeros(param[1]))  # Shape: [attention_size]
        #         context_w = nn.Parameter(torch.ones(1, param[1]))  # Shape: [attention_size, 1]
        #         nn.init.kaiming_normal_(attention_w)  # Initialize weights
        #         self.vars.append(attention_w)
        #         self.vars.append(attention_b)
        #         self.vars.append(context_w)
        #     else:
        #         raise NotImplementedError

        # Append locomotion mode head parameters
        for name, param in self.loc_head_config:
            if name == 'linear':
                w = nn.Parameter(torch.ones(*param)) # weight dimensions
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) # bias dimensions
            elif name == 'relu':
                continue
            else:
                raise NotImplementedError

        # Add incline regression head
        for name, param in self.incline_head_config:
            if name == 'linear':
                w = nn.Parameter(torch.ones(*param)) # weight dimensions
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) # bias dimensions
            elif name == 'relu':
                continue
            else:
                raise NotImplementedError

        # Add gait classification head
        for name, param in self.gait_head_config:
            if name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  # Bias
            elif name == 'relu':
                continue
            else:
                raise NotImplementedError

        # # Output heads for the different tasks
        # if last_layer_size is not None:
        #
        #     # # LSTM for incline detection
        #     # self.incline_lstm = nn.LSTM(input_size=last_layer_size, hidden_size=128, num_layers=1, batch_first=True)
        #     # self.incline_regression_head = nn.Linear(128, 1)  # Incline regression
        #
        #     self.gait_classification_head = nn.Linear(last_layer_size, 4)  # Gait phase classification (4 classes)
        #     # self.incline_regression_head = nn.Linear(last_layer_size, 1)
        #
        #     # self.gait_classification_head = nn.Sequential(
        #     #     nn.Linear(last_layer_size, 32),
        #     #     nn.ReLU(),
        #     #     nn.Linear(32, 32),
        #     #     nn.ReLU(),
        #     #     nn.Linear(32, 4)
        #     # )
        #
        #     self.incline_regression_head = nn.Sequential(
        #         nn.Linear(last_layer_size, 32),
        #         nn.ReLU(),
        #         # nn.Linear(32, 32),
        #         # nn.ReLU(),
        #         nn.Linear(32, 1)
        #     )


    def extra_repr(self):
        info = "Base Network:\n"

        for name, param in self.config:
            if name == 'conv1d':
                tmp = 'conv1d:(ch_in:%d, ch_out:%d, k:%d, stride:%d, padding:%d, dilation:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'dropout']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        # Reduced layer configuration
        # info += "\nAttention layer:\n"
        #
        # for name, param in self.attention_layer_config:
        #     if name == 'attention':
        #         tmp = 'attention:(in:%d, attention_size:%d)' % (param[0], param[1])
        #         info += tmp + '\n'
        #
        #     elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'dropout']:
        #         tmp = name + ':' + str(tuple(param))
        #         info += tmp + '\n'
        #     else:
        #         raise NotImplementedError

        # Locomotion classification head configuration
        info += "\nLocomotion Classification Head:\n"

        for name, param in self.loc_head_config:
            if name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'dropout']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        # Incline regression head configuration
        info += "\nIncline Regression Head:\n"

        for name, param in self.incline_head_config:
            if name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn',
                          'dropout']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        # Gait classification head configuration
        info += "\nGait Classification Head:\n"

        for name, param in self.gait_head_config:
            if name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'dropout']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError
        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, input_channel, setsz]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv1d(x, w, b, stride=param[3], padding=param[4], dilation=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'dropout':
                # Apply dropout with the specified rate from param
                x = F.dropout(x, p=param[0], training=self.training)
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # # attention layer
        # attention_layer_output = x
        # for name, param in self.attention_layer_config:
        #     if name == 'attention':
        #         attention_w, attention_b = vars[idx], vars[idx + 1]
        #         context_w = vars[idx + 2]
        #         # Compute attention scores
        #         attention_scores = F.linear(attention_layer_output, attention_w, attention_b)
        #         attention_scores = torch.tanh(attention_scores)
        #         # Normalize scores into weights
        #         attention_weights = F.softmax(F.linear(attention_scores, context_w), dim=1)
        #         # Compute weighted sum
        #         attention_layer_output = (attention_weights * attention_layer_output.unsqueeze(1)).sum(dim=1)
        #         idx += 3

        # Locomotion classification
        loc_output = x
        for name, param in self.loc_head_config:
            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                loc_output = F.linear(loc_output, w, b)
                idx += 2
            elif name == 'relu':
                loc_output = F.relu(loc_output, inplace=True)

        # Concatenate locomotion output with original features
        # loc_features = F.softmax(loc_output, dim=1) # Softmax for probabilities
        loc_features = torch.argmax(loc_output, dim=1, keepdim=True).float()
        x_loc = torch.cat([x, loc_features], dim=1)
        # x_loc = torch.cat([x, loc_output], dim=1)
        # x_loc = torch.cat([attention_layer_output, loc_output], dim=1)

        # Incline regression head
        incline_output = x_loc
        for name, param in self.incline_head_config:
            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                incline_output = F.linear(incline_output, w, b)
                idx += 2
            elif name == 'relu':
                incline_output = F.relu(incline_output, inplace=True)

        # Gait classification head
        # gait_output = x
        gait_output = x_loc
        for name, param in self.gait_head_config:
            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                gait_output = F.linear(gait_output, w, b)
                idx += 2
            elif name == 'relu':
                gait_output = F.relu(gait_output, inplace=True)

        # make sure variable is used properly
        assert idx == len(vars), f"Not all vars were used! {idx}/{len(vars)}"
        assert bn_idx == len(self.vars_bn), f"Not all bn_vars were used! {bn_idx}/{len(self.vars_bn)}"

        return loc_output, gait_output, incline_output

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars