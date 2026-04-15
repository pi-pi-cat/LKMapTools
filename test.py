import cv2
print("OpenCV 版本:", cv2.__version__)
print("CUDA 数量:", cv2.cuda.getCudaEnabledDeviceCount())

# 检查是否真的能用（尝试创建一个 CUDA 矩阵）
try:
    gpu_frame = cv2.cuda_GpuMat()
    print("CUDA 模块初始化成功！")
except Exception as e:
    print("CUDA 模块虽然存在，但初始化失败，可能是 DLL 没找对位置:", e)