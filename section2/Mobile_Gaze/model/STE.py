import torch

class BinarizeSTE_origin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 直接对输入进行二值化，输出为 0 或 1
        binary_output = (input >= 0).float()
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播中，梯度直接传递
        grad_input = grad_output.clone()
        return grad_input




class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 使用 tanh 函数进行近似
        output = torch.tanh(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 使用 tanh 的导数传递梯度
        grad_input = grad_output * (1 - torch.tanh(input) ** 2)
        return grad_input


class BinarizeSTEWithL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 在前向传播中，返回经过 sigmoid 离散化后的结果
        sigmoid_output = torch.sigmoid(input)
        binary_output = (sigmoid_output > 0.5).float()
        ctx.save_for_backward(sigmoid_output)  # 保存前向传播时的 sigmoid 结果以用于反向传播
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_output, = ctx.saved_tensors
        # L2 惩罚项梯度的部分

        scale = 0.9
        l2_penalty_gradient = 2 * (sigmoid_output - 0.5)
        # 根据 sigmoid_output 调整梯度，增加向 0.5 的靠拢
        adjustment_factor = torch.where(sigmoid_output > 0.5, -l2_penalty_gradient, l2_penalty_gradient)
        # 最终的梯度是原始的 grad_output 加上调整因子后的值
        adjusted_grad = scale * (grad_output + adjustment_factor)
        # 返回调整后的梯度
        return adjusted_grad