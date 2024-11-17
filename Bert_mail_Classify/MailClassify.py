# 导入必要的库
import pandas as pd  # 用于数据处理
import torch  # 用于深度学习模型
from sklearn.metrics import classification_report  # 用于评估分类模型的性能
from transformers import BertTokenizer, BertForSequenceClassification  # 用于加载BERT模型和分词器
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback  # 用于模型训练和早停策略

# 自定义数据集类
class EmailDataset(torch.utils.data.Dataset):
    # 初始化函数，接收编码后的数据并设置标签
    def __init__(self, encodings):
        self.encodings = encodings  # 输入特征
        self.labels = encodings.pop('labels', None)  # 标签（如果有的话）

    # 获取特定索引的数据项
    def __getitem__(self, idx):
        # 将编码后的特征按索引取出，组成一个字典
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]  # 如果有标签，将其添加到数据项中
        return item

    # 返回数据集的长度
    def __len__(self):
        return len(self.encodings['input_ids'])  # 数据长度基于input_ids特征

# 加载训练和测试数据
def load_data():
    # 从CSV文件读取训练集和测试集数据
    train_data = pd.read_csv('mail_data_train.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv('mail_data_test_nolable.csv', encoding='ISO-8859-1')
    return train_data, test_data

# 预处理邮件内容，去除非字母数字字符
def preprocess_content(content):
    # 仅保留字母数字的单词并通过空格连接
    return ' '.join(word for word in content.split() if word.isalnum())

# 将文本数据编码为BERT模型所需的输入格式
def encode_data(tokenizer, texts, labels=None):
    # 对文本进行分词和编码，支持自动填充和截断
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    # 如果有标签，将其添加到编码字典中
    if labels is not None:
        encoding['labels'] = torch.tensor(labels)
    return encoding

# 主函数，包含数据处理、模型训练和预测等步骤
def main():
    # 加载训练和测试数据
    train_data, test_data = load_data()

    # 将训练数据拆分为训练集和验证集
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_data, test_size=0.1)

    # 对训练、验证和测试数据的邮件内容进行预处理
    train_data['Processed_Message'] = train_data['Message'].apply(preprocess_content)
    test_data['Processed_Message'] = test_data['Message'].apply(preprocess_content)
    val_data['Processed_Message'] = val_data['Message'].apply(preprocess_content)

    # 使用本地保存的BERT模型和分词器
    local_model_path = 'Bert_uncased'  # 本地模型文件夹路径
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=len(train_data['Category'].unique()))

    # 编码训练数据并创建EmailDataset数据集
    train_encodings = encode_data(tokenizer, train_data['Processed_Message'].tolist(), train_data['Category'].tolist())
    train_dataset = EmailDataset(train_encodings)

    # 编码验证数据并创建EmailDataset数据集
    val_encodings = encode_data(tokenizer, val_data['Processed_Message'].tolist(), val_data['Category'].tolist())
    val_dataset = EmailDataset(val_encodings)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',  # 模型保存路径
        num_train_epochs=10,  # 设置训练周期数
        per_device_train_batch_size=64,  # 每个设备的批量大小
        learning_rate=5e-5,  # 设置学习率
        logging_dir='./logs',  # 日志保存路径
        logging_steps=10,  # 每隔10步记录一次日志
        evaluation_strategy='epoch',  # 在每个epoch结束时进行验证
        save_strategy='epoch',  # 在每个epoch结束时保存模型
        load_best_model_at_end=True,  # 在训练结束后加载最佳模型
        eval_steps=100,  # 每100步进行一次验证
        metric_for_best_model="eval_loss",  # 使用验证损失作为早停的评估标准
        greater_is_better=False,  # 评估标准为越小越好
    )

    # 创建Trainer实例，配置模型、训练参数、数据集和早停策略
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 早停策略：若3个周期内无提升则停止训练
    )

    # 训练模型
    trainer.train()

    # 对测试集进行编码并生成数据集
    test_encodings = encode_data(tokenizer, test_data['Processed_Message'].tolist())
    test_dataset = EmailDataset(test_encodings)

    # 预测测试集类别
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)  # 获取预测标签

    # 保存预测结果到CSV文件
    test_data['Category'] = predicted_labels
    submission = test_data[['id', 'Category']]
    submission.to_csv('sample_submission.csv', index=False)
    print("预测结果已保存到 sample_submission.csv")

    # 打印训练集上的分类报告
    print(classification_report(train_data['Category'], trainer.predict(train_dataset).predictions.argmax(-1)))

# 执行主函数
if __name__ == '__main__':
    main()
