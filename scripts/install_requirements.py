"""
安装必要的Python包
"""

REQUIRED_PACKAGES = [
    "pandas>=1.3.0",
    "numpy>=1.20.0", 
    "scikit-learn>=1.0.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "lightgbm>=3.0.0",
    "catboost>=1.0.0"
]

OPTIONAL_PACKAGES = [
    "jupyter>=1.0.0",
    "plotly>=5.0.0",
    "yfinance>=0.1.70"
]

def install_packages():
    import subprocess
    import sys
    
    print("安装必要的包...")
    
    for package in REQUIRED_PACKAGES:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
    
    print("\n可选包安装（可跳过）...")
    for package in OPTIONAL_PACKAGES:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败，跳过")

if __name__ == "__main__":
    install_packages()
