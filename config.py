"""
项目配置文件
集中管理所有路径和配置参数
"""
import os

# 获取项目根目录（自动检测，无需硬编码）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 目录路径配置
# ==========================================
PATHS = {
    # 源码目录
    "src": os.path.join(PROJECT_ROOT, "src"),
    
    # 数据目录
    "data_raw": os.path.join(PROJECT_ROOT, "data", "raw"),
    "data_processed": os.path.join(PROJECT_ROOT, "data", "processed"),
    
    # 输出目录
    "figures": os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs", "results"),
    
    # 脚本目录
    "scripts": os.path.join(PROJECT_ROOT, "scripts"),
}

# ==========================================
# 文件路径配置
# ==========================================
FILES = {
    # 原始数据
    "stroke_data": os.path.join(PATHS["data_raw"], "healthcare-dataset-stroke-data.csv"),
    
    # 模型和处理后的数据
    "model": os.path.join(PATHS["data_processed"], "model_lgb.pkl"),
    "test_data": os.path.join(PATHS["data_processed"], "test_data.pkl"),
    "normal_data": os.path.join(PATHS["data_processed"], "normal_data.pkl"),
    "config_pkl": os.path.join(PATHS["data_processed"], "config.pkl"),
}

# ==========================================
# 模型参数配置
# ==========================================
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_neighbors": 5,  # KNN 邻居数
}

# ==========================================
# 辅助函数
# ==========================================
def ensure_dirs():
    """确保所有必要的目录都存在"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

def get_path(key):
    """获取路径的便捷方法"""
    return PATHS.get(key) or FILES.get(key)

