import sys
print("Python 路径:")
for path in sys.path:
    if 'site-packages' in path:
        print(f"  {path}")

# 测试导入
try:
    import utils
    print("✅ 成功导入 utils 包")
    
    from utils.model.base import ModernModule
    print("✅ 成功导入 ModernModule")
    
    from utils.dataset import DataDownloader
    print("✅ 成功导入 DataDownloader")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")