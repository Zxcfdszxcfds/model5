import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, color, exposure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm

# ---------------------- 配置 ----------------------
st.set_page_config(page_title="模式识别A5作业平台", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 数据加载 ----------------------
@st.cache_resource
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()

# ---------------------- 模块1：HOG+词袋+SVM ----------------------
def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_gray = color.rgb2gray(img) if len(img.shape)==3 else img
        features = hog(img_gray, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

def build_bovw(features, n_clusters=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

def get_bovw_histogram(features, kmeans):
    labels = kmeans.predict(features)
    hist, _ = np.histogram(labels, bins=range(kmeans.n_clusters+1))
    return hist / hist.sum()

# ---------------------- 模块2：反向传播演示 ----------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_backprop_demo():
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
    model = SimpleNet()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_history = []
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history

# ---------------------- 模块3：LeNet-5训练 ----------------------
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_lenet(model, train_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
    return loss_history

def test_lenet(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

# ---------------------- 模块4：ResNet性能对比 ----------------------
def test_resnet(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 适配ResNet输入（3通道）
            data = data.repeat(1,3,1,1).to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

# ---------------------- Streamlit界面 ----------------------
st.title("📊 模式识别A5作业平台")
tab1, tab2, tab3, tab4 = st.tabs([
    "1. HOG+词袋+SVM",
    "2. 反向传播演示",
    "3. LeNet-5训练测试",
    "4. ResNet性能对比"
])

# ---------------------- 模块1：HOG+词袋+SVM ----------------------
with tab1:
    st.header("HOG+词袋+SVM图像分类")
    st.subheader("使用MNIST子集演示HOG特征提取与SVM分类")
    if st.button("运行HOG+词袋+SVM", key="run_hog"):
        with st.spinner("处理中..."):
            # 取MNIST子集
            X_train = train_dataset.data[:1000].numpy()
            y_train = train_dataset.targets[:1000].numpy()
            X_test = test_dataset.data[:200].numpy()
            y_test = test_dataset.targets[:200].numpy()
            
            # 提取HOG特征
            train_hog = extract_hog_features(X_train)
            test_hog = extract_hog_features(X_test)
            
            # 构建词袋
            kmeans = build_bovw(train_hog, n_clusters=30)
            
            # 生成词袋直方图
            X_train_bovw = np.array([get_bovw_histogram(f.reshape(1,-1), kmeans) for f in train_hog])
            X_test_bovw = np.array([get_bovw_histogram(f.reshape(1,-1), kmeans) for f in test_hog])
            
            # 训练SVM
            svm = SVC(kernel='linear')
            svm.fit(X_train_bovw, y_train)
            y_pred = svm.predict(X_test_bovw)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"测试集准确率: {acc:.2f}")
            
            # 可视化HOG特征
            img = X_train[0]
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8,8),
                               cells_per_block=(2,2), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title("原图")
            axes[1].imshow(hog_image_rescaled, cmap='gray')
            axes[1].set_title("HOG特征图")
            st.pyplot(fig)

# ---------------------- 模块2：反向传播演示 ----------------------
with tab2:
    st.header("反向传播演示（XOR问题）")
    if st.button("运行反向传播训练", key="run_bp"):
        with st.spinner("训练中..."):
            model, loss_history = train_backprop_demo()
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(loss_history)
            ax.set_title("反向传播损失曲线")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            st.pyplot(fig)
            
            # 测试结果
            X_test = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X_test)
            st.subheader("XOR测试结果")
            for x, y in zip(X_test, outputs):
                st.write(f"输入: {x.numpy()} → 输出: {y.item():.4f}")

# ---------------------- 模块3：LeNet-5训练测试 ----------------------
with tab3:
    st.header("LeNet-5手写数字识别")
    epochs_lenet = st.slider("训练轮数", 1, 5, 3, key="epochs_lenet")
    if st.button("训练LeNet-5", key="run_lenet"):
        with st.spinner("训练中..."):
            model = LeNet5().to(device)
            loss_history = train_lenet(model, train_loader, epochs=epochs_lenet)
            acc = test_lenet(model, test_loader)
            
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(loss_history)
            ax.set_title("LeNet-5训练损失曲线")
            st.pyplot(fig)
            
            st.success(f"测试集准确率: {acc:.2f}%")
            
            # 测试样本
            model.eval()
            test_samples, test_labels = next(iter(test_loader))
            test_samples = test_samples[:5].to(device)
            with torch.no_grad():
                preds = model(test_samples).argmax(dim=1)
            
            fig, axes = plt.subplots(1,5, figsize=(15,3))
            for i in range(5):
                axes[i].imshow(test_samples[i,0].cpu().numpy(), cmap='gray')
                axes[i].set_title(f"真实:{test_labels[i]}\n预测:{preds[i]}")
                axes[i].axis('off')
            st.pyplot(fig)

# ---------------------- 模块4：ResNet性能对比 ----------------------
with tab4:
    st.header("不同深度ResNet性能对比")
    if st.button("对比ResNet18/34/50", key="run_resnet"):
        with st.spinner("测试中..."):
            models = {
                "ResNet18": resnet18(pretrained=False, num_classes=10).to(device),
                "ResNet34": resnet34(pretrained=False, num_classes=10).to(device),
                "ResNet50": resnet50(pretrained=False, num_classes=10).to(device)
            }
            results = {}
            for name, model in models.items():
                acc = test_resnet(model, test_loader)
                results[name] = acc
            
            st.subheader("测试准确率对比")
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(results.keys(), results.values(), color=['blue','green','orange'])
            ax.set_ylabel("准确率 (%)")
            st.pyplot(fig)
            
            st.write("### 结果说明")
            st.markdown("""
            - 更深的ResNet理论上拟合能力更强，但需要更多训练数据和计算资源
            - 本次未预训练，浅层模型（ResNet18/34）在小数据上表现更稳定
            """)

st.markdown("---")
st.caption("模式识别与图像处理 - A5作业平台")
