import re
import os
import matplotlib.pyplot as plt
from datetime import datetime

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extract_metrics(lines):
    # 修正后的正则表达式
    pattern = re.compile(
        # r'Time:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ Test set: Average loss: (\d+\.\d+), Accuracy: \d+/\d+ \((\d+\.\d+)%\)'
        r'Time:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ Test set: Average loss: (\d+\.\d+), Accuracy: \d+(?:\.\d+)?/\d+(?:\.\d+)? \((\d+\.\d+)%\)'
    )

    losses = []
    accuracies = []

    for line in lines:
        match = pattern.search(line)
        if match:
            loss, accuracy = match.groups()
            losses.append(float(loss))
            accuracies.append(float(accuracy))

    return losses, accuracies

def plot_loss(losses, save_dir=None):
    # 创建指定尺寸的图形（单位：英寸）
    plt.figure(figsize=(6.40, 4.80))  # 512像素 ÷ 100DPI = 5.12英寸
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-', label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # ==== 关键修改：先保存后显示 ====
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"loss_plot_{timestamp}.png")
        # 关键参数设置
        plt.savefig(
            save_path,
            dpi=100,  # 每英寸点数
            bbox_inches='tight',  # 自动裁剪白边
            pad_inches=0.1,  # 保留少量边距
            metadata={'Software': 'Matplotlib'},  # 可选元数据
            facecolor='white'  # 设置背景为纯白
        )
        print(f"Saved loss plot to: {save_path}")

    plt.show()  # 后显示
    plt.close()  # 关闭图形释放内存


def plot_accuracy(accuracies, save_dir=None):
    # 创建指定尺寸的图形（单位：英寸）
    plt.figure(figsize=(6.40, 4.80))  # 512像素 ÷ 100DPI = 5.12英寸
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, 'r-', label='Test Accuracy (%)')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"accuracy_plot_{timestamp}.png")
        # 关键参数设置
        plt.savefig(
            save_path,
            dpi=100,  # 每英寸点数
            bbox_inches='tight',  # 自动裁剪白边
            pad_inches=0.1,  # 保留少量边距
            metadata={'Software': 'Matplotlib'},  # 可选元数据
            facecolor='white'  # 设置背景为纯白
        )
        print(f"Saved accuracy plot to: {save_path}")

    plt.show()  # 后显示
    plt.close()


if __name__ == "__main__":

    # 新增保存目录配置
    output_dir = r"C:\Users\86130\Desktop\srnet训练结果\实验\M-A无OT"  # 可修改为任意路径
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    file_path = r"C:\Users\86130\Desktop\srnet训练结果\实验\M-A无OT\B-A_S_test-acc.txt"
    lines = read_txt_file(file_path)
    losses, accuracies = extract_metrics(lines)

    if len(losses) == 0 or len(accuracies) == 0:
        print("未提取到数据，请检查正则表达式或文件内容！")
    else:
        # 计算平均值
        average_loss = sum(losses) / len(losses)
        average_accuracy = sum(accuracies) / len(accuracies)

        # 打印结果（保留4位小数）
        print(f"Average Test Loss: {average_loss:.4f}")
        print(f"Average Test Accuracy: {average_accuracy:.2f}%")

        # 修改绘图调用方式
        plot_loss(losses, save_dir=output_dir)  # 传递保存路径
        plot_accuracy(accuracies, save_dir=output_dir)