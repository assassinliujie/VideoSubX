"""
VideoSubX MFA 安装脚本
在当前 conda 环境中安装 Montreal Forced Aligner

注意：
- 必须在已激活的 conda 环境中运行（如 videosubx）
- 使用 conda-forge 安装 MFA 及其依赖（包括 Kaldi）
- 自动读取 config.yaml 中的模型配置并下载
"""

import os
import sys
import subprocess
import shutil

# Windows 编码修复：设置 UTF-8 编码以避免 GBK 相关错误
if sys.platform == 'win32':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_conda_env():
    """检查是否在 conda 环境中"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("❌ 未检测到 conda 环境！")
        print("\n请先激活您的 conda 环境，例如:")
        print("   conda activate videosubx")
        return None
    
    env_name = os.path.basename(conda_prefix)
    print(f"✅ 当前 conda 环境: {env_name}")
    return env_name

def run_conda_install(*packages):
    """在当前环境中运行 conda install"""
    cmd = ['conda', 'install', '-c', 'conda-forge', '-y'] + list(packages)
    print(f"   运行: conda install -c conda-forge {' '.join(packages)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_pip_uninstall(*packages):
    """卸载 pip 包"""
    cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y'] + list(packages)
    subprocess.run(cmd, capture_output=True)

def fix_dependencies():
    """修复 MFA 依赖问题
    
    MFA 的 kalpy 模块需要 conda 版本的 numpy，
    如果 numpy 是 pip 安装的会导致二进制不兼容。
    """
    print("   🔧 检查并修复依赖...")
    
    # 检查 numpy 是否是 pip 安装的
    result = subprocess.run(
        ['conda', 'list', 'numpy'],
        capture_output=True, text=True
    )
    
    if 'pypi' in result.stdout:
        print("   ⚠️ 检测到 pip 版本的 numpy，正在替换为 conda 版本...")
        run_pip_uninstall('numpy')
        run_conda_install('numpy=1.26.4')
        print("   ✅ numpy 已修复")
    else:
        print("   ✅ numpy 版本兼容")

def fix_ffmpeg_conflict():
    """修复 ffmpeg DLL 冲突问题
    
    MFA 安装会带入不兼容的 avcodec DLL，与系统 ffmpeg 冲突。
    删除 conda 环境中的冲突文件，让系统使用用户自己安装的 ffmpeg。
    """
    print("   🔧 检查 ffmpeg 冲突...")
    
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if not conda_prefix:
        return
    
    lib_bin = os.path.join(conda_prefix, 'Library', 'bin')
    if not os.path.exists(lib_bin):
        return
    
    # 需要删除的冲突文件模式
    conflict_patterns = [
        'avcodec*.dll', 'avformat*.dll', 'avutil*.dll',
        'swscale*.dll', 'swresample*.dll', 'avdevice*.dll',
        'avfilter*.dll', 'ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe'
    ]
    
    deleted_count = 0
    import glob
    for pattern in conflict_patterns:
        for file in glob.glob(os.path.join(lib_bin, pattern)):
            try:
                os.remove(file)
                deleted_count += 1
            except Exception:
                pass
    
    if deleted_count > 0:
        print(f"   ✅ 已清理 {deleted_count} 个冲突文件，使用系统 ffmpeg")
    else:
        print("   ✅ 无 ffmpeg 冲突")

def get_mfa_config():
    """从 config.yaml 读取 MFA 配置"""
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        mfa_config = config.get('mfa', {})
        model_dir = config.get('model_dir', './_model_cache')
        
        return {
            'acoustic_model': mfa_config.get('acoustic_model', 'english_mfa'),
            'dictionary': mfa_config.get('dictionary', 'english_mfa'),
            'model_dir': os.path.abspath(model_dir)
        }
    except Exception as e:
        print(f"⚠️ 读取 config.yaml 失败: {e}")
        print("   使用默认配置: english_mfa")
        return {
            'acoustic_model': 'english_mfa',
            'dictionary': 'english_mfa',
            'model_dir': os.path.abspath('./_model_cache')
        }

def run_mfa_command(mfa_args):
    """运行 MFA 命令"""
    cmd = ['mfa'] + mfa_args
    print(f"   运行: mfa {' '.join(mfa_args)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        # 只显示关键错误，忽略常见警告
        if 'error' in result.stderr.lower():
            print(f"⚠️ MFA 命令警告: {result.stderr[:300]}")
    return result.returncode == 0

def check_mfa_installed():
    """检查 MFA 是否已安装"""
    try:
        result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        # mfa 命令不存在
        pass
    except Exception:
        pass
    return None

def main():
    """主安装流程"""
    print("\n" + "="*60)
    print("   VideoSubX - MFA 安装程序")
    print("   Montreal Forced Aligner 声学对齐工具")
    print("="*60 + "\n")
    
    # 1. 检查 conda 环境
    print("📦 步骤 1/4: 检查 conda 环境...")
    env_name = check_conda_env()
    if not env_name:
        return False
    
    # 2. 安装 MFA
    print("\n📦 步骤 2/4: 安装 Montreal Forced Aligner...")
    
    # 先检查是否已安装
    version = check_mfa_installed()
    if version:
        print(f"✅ MFA 已安装，版本: {version}")
    else:
        print("   正在安装 MFA（这可能需要几分钟）...")
        success = run_conda_install('montreal-forced-aligner')
        if not success:
            print("❌ MFA 安装失败")
            print("\n请尝试手动安装:")
            print("   conda install -c conda-forge montreal-forced-aligner")
            return False
        print("✅ MFA 安装成功")
    
    # 修复依赖（确保 numpy 是 conda 版本以兼容 kalpy）
    fix_dependencies()
    
    # 修复 ffmpeg 冲突（删除与系统 ffmpeg 冲突的 DLL）
    fix_ffmpeg_conflict()
    
    # 3. 读取配置并下载模型
    print("\n📦 步骤 3/4: 读取配置并下载模型...")
    config = get_mfa_config()
    print(f"   声学模型: {config['acoustic_model']}")
    print(f"   发音词典: {config['dictionary']}")
    print(f"   模型目录: {config['model_dir']}")
    
    # 创建模型目录
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # 下载声学模型
    print(f"\n   📥 下载声学模型: {config['acoustic_model']}...")
    run_mfa_command(['model', 'download', 'acoustic', config['acoustic_model']])
    
    # 下载词典
    print(f"\n   📥 下载发音词典: {config['dictionary']}...")
    run_mfa_command(['model', 'download', 'dictionary', config['dictionary']])
    
    # 4. 验证安装
    print("\n📦 步骤 4/4: 验证安装...")
    version = check_mfa_installed()
    if version:
        print(f"✅ MFA 安装验证成功！版本: {version}")
    else:
        print("⚠️ MFA 安装可能不完整，请检查输出")
    
    print("\n" + "="*60)
    print("   安装完成！")
    print("="*60)
    print("\n使用方法:")
    print("   1. 在 config.yaml 中设置 mfa.enabled: true")
    print("   2. 正常运行项目，MFA 会自动优化时间轴")
    print("\n如需更换语言模型，请修改 config.yaml 中的:")
    print("   mfa.acoustic_model 和 mfa.dictionary")
    print("   然后重新运行此脚本下载对应模型")
    print("\n可用模型列表: https://mfa-models.readthedocs.io/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
