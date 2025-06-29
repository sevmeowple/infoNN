# DeepLearning项目打包脚本
# 作者: SevFoxie
# 日期: 2025-06-27

param(
    [string]$OutputPath = ".\dist",
    [string]$PackageName = "deeplearning_study",
    [switch]$IncludeData = $false,
    [switch]$Clean = $false
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取脚本所在目录
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptPath

Write-Host "🚀 开始打包 DeepLearning 项目..." -ForegroundColor Green

# 清理旧的打包文件
if ($Clean -and (Test-Path $OutputPath)) {
    Write-Host "🧹 清理旧的打包文件..." -ForegroundColor Yellow
    Remove-Item $OutputPath -Recurse -Force
}

# 创建输出目录
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath | Out-Null
}

# 定义要包含的文件和目录
$IncludeItems = @(
    "src",
    "notebooks",
    "pyproject.toml",
    "README.md",
    "uv.lock",
    ".python-version"
)

# 定义要排除的模式
$ExcludePatterns = @(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".coverage",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    "lightning_logs",
    "*.ckpt"
)

# 如果包含数据，添加数据目录
if ($IncludeData) {
    $IncludeItems += "data"
    Write-Host "📊 包含数据文件..." -ForegroundColor Cyan
} else {
    Write-Host "⚠️  跳过数据文件 (使用 -IncludeData 包含)" -ForegroundColor Yellow
}

# 创建临时目录
$TempDir = Join-Path $OutputPath "temp_$PackageName"
if (Test-Path $TempDir) {
    Remove-Item $TempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "📦 复制项目文件..." -ForegroundColor Cyan

# 复制文件
foreach ($Item in $IncludeItems) {
    if (Test-Path $Item) {
        $DestPath = Join-Path $TempDir $Item
        
        if (Test-Path $Item -PathType Container) {
            # 复制目录
            Write-Host "  📁 复制目录: $Item" -ForegroundColor Gray
            Copy-Item $Item $DestPath -Recurse -Force
            
            # 清理不需要的文件
            foreach ($Pattern in $ExcludePatterns) {
                Get-ChildItem $DestPath -Recurse -Force | 
                Where-Object { $_.Name -like $Pattern -or $_.PSIsContainer -and $_.Name -like $Pattern } |
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            }
        } else {
            # 复制文件
            Write-Host "  📄 复制文件: $Item" -ForegroundColor Gray
            Copy-Item $Item $DestPath -Force
        }
    } else {
        Write-Host "  ⚠️  文件不存在: $Item" -ForegroundColor Yellow
    }
}

# 创建项目信息文件
$ProjectInfo = @{
    "project_name" = $PackageName
    "package_date" = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "version" = "1.0.0"
    "author" = "SevFoxie"
    "description" = "Deep Learning Study Project"
    "python_version" = (Get-Content ".python-version" -ErrorAction SilentlyContinue)
    "include_data" = $IncludeData
}

$ProjectInfo | ConvertTo-Json -Depth 2 | Out-File (Join-Path $TempDir "package_info.json") -Encoding UTF8

# 创建安装说明
$InstallInstructions = @"
# Deep Learning Study Project

## 安装说明

1. 确保安装了 Python $(if($ProjectInfo.python_version) { $ProjectInfo.python_version } else { "3.8+" })
2. 安装 uv 包管理器:
   ```
   pip install uv
   ```
3. 安装依赖:
   ```
   uv sync
   ```
4. 激活虚拟环境:
   ```
   uv venv
   .venv\Scripts\activate  # Windows
   ```

## 项目结构

- `src/` - 源代码
- `notebooks/` - Jupyter notebooks
- `data/` - 数据文件 $(if(-not $IncludeData) { "(未包含，需要单独下载)" } else { "" })

## 运行

```bash
cd notebooks/learn
jupyter lab
```

打包时间: $($ProjectInfo.package_date)
"@

$InstallInstructions | Out-File (Join-Path $TempDir "INSTALL.md") -Encoding UTF8

# 创建压缩包
$ZipPath = Join-Path $OutputPath "$PackageName-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
Write-Host "🗜️  创建压缩包: $ZipPath" -ForegroundColor Green

# 确保输出目录存在
$ZipDir = Split-Path $ZipPath -Parent
if (-not (Test-Path $ZipDir)) {
    New-Item -ItemType Directory -Path $ZipDir -Force | Out-Null
}

# 如果ZIP文件已存在，先删除
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

$ZipCreated = $false

try {
    # 使用 .NET 方法创建 ZIP
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($TempDir, $ZipPath)
    $ZipCreated = $true
    Write-Host "✅ 使用.NET方法创建ZIP成功" -ForegroundColor Green
} catch {
    Write-Host "❌ 创建ZIP文件失败，尝试备用方法..." -ForegroundColor Red
    
    # 备用方法：使用 Compress-Archive
    try {
        $Items = Get-ChildItem $TempDir
        Compress-Archive -Path $Items.FullName -DestinationPath $ZipPath -Force
        $ZipCreated = $true
        Write-Host "✅ 使用备用方法创建ZIP成功" -ForegroundColor Green
    } catch {
        Write-Host "❌ 所有打包方法都失败了: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# 删除重复的ZIP创建代码（第185行）

# 清理临时目录
Remove-Item $TempDir -Recurse -Force

# 显示结果
if ($ZipCreated -and (Test-Path $ZipPath)) {
    $ZipInfo = Get-Item $ZipPath
    Write-Host "✅ 打包完成!" -ForegroundColor Green
    Write-Host "📦 文件位置: $($ZipInfo.FullName)" -ForegroundColor Cyan
    Write-Host "📊 文件大小: $([math]::Round($ZipInfo.Length / 1MB, 2)) MB" -ForegroundColor Cyan
} else {
    Write-Host "❌ 打包失败!" -ForegroundColor Red
    exit 1
}
# 创建快速安装脚本
$QuickInstall = @"
# 快速安装脚本
Write-Host "🚀 安装 Deep Learning Study Project..." -ForegroundColor Green

# 检查 Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 请先安装 Python" -ForegroundColor Red
    exit 1
}

# 检查 uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 安装 uv..." -ForegroundColor Yellow
    pip install uv
}

# 安装依赖
Write-Host "📚 安装依赖..." -ForegroundColor Yellow
uv sync

Write-Host "✅ 安装完成!" -ForegroundColor Green
Write-Host "🎓 运行: cd notebooks/learn && jupyter lab" -ForegroundColor Cyan
"@

$QuickInstall | Out-File (Join-Path $OutputPath "quick_install.ps1") -Encoding UTF8

Write-Host "🛠️  已创建快速安装脚本: $(Join-Path $OutputPath 'quick_install.ps1')" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 使用方法:" -ForegroundColor White
Write-Host "  基本打包: .\package.ps1" -ForegroundColor Gray
Write-Host "  包含数据: .\package.ps1 -IncludeData" -ForegroundColor Gray
Write-Host "  清理重建: .\package.ps1 -Clean" -ForegroundColor Gray
Write-Host "  自定义路径: .\package.ps1 -OutputPath 'C:\MyPackages'" -ForegroundColor Gray