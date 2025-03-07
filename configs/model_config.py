"""
模型配置文件
"""

class ModelConfig:
    # 模型架构配置
    input_channels = 10  # 输入通道数
    hidden_channels = 64  # 隐藏层通道数
    output_channels = 2  # 输出通道数（海流预测：u, v分量）
    num_layers = 4  # 模型层数
    
    # 训练配置
    batch_size = 32
    learning_rate = 1e-4
    max_epochs = 100
    
    # 数据配置
    input_size = (224, 224)  # 输入数据大小
    normalize = True  # 是否进行数据标准化
    
    # 预训练模型配置
    pretrained_model_path = "checkpoints/ocp_model_latest.pth"
    
    # 设备配置
    device = "cuda"  # or "cpu"
    
    # 日志配置
    log_dir = "logs"
    save_interval = 10  # 每多少个epoch保存一次模型
    
    def __str__(self):
        """打印配置信息"""
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items()
                        if not k.startswith("_")) 