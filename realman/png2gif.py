from PIL import Image
import os

def create_gif_from_images(image_folder, output_gif_path, duration=200, loop=0):
    """
    将文件夹中的图片合并成一个GIF动画
    :param image_folder: 图片文件夹路径
    :param output_gif_path: 输出的GIF文件路径
    :param duration: 每帧的显示时间（毫秒）
    :param loop: 循环次数，0表示无限循环
    """
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    images.sort()  # 按文件名排序

    if not images:
        print("文件夹中没有图片文件！")
        return

    # 打开所有图片并存入列表
    frames = []
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = Image.open(image_path)
        frames.append(frame)

    # 保存为GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF已保存到：{output_gif_path}")

# 示例用法
image_folder = "/home/ubuntu/project/DOA/手撕代码/result"  # 替换为你的图片文件夹路径
output_gif_path = "output.gif"  # 输出的GIF文件路径
create_gif_from_images(image_folder, output_gif_path, duration=200, loop=0)