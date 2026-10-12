# LaRSE-Expand 环境依赖说明

本文记录 LaRSE-Expand 的参考环境，依据已有环境的 Conda 和 pip 包清单整理。版本按原始记录保留，尚未验证从零安装与完整运行流程。

## 基础环境

| 项目 | 版本 / 说明 |
| --- | --- |
| 平台 | Linux x86_64（根据 Conda 工具链记录） |
| Python | 3.7.12，CPython，ABI 为 cp37m |
| 环境管理 | Conda，版本未提供 |
| Conda 软件源 | conda-forge |
| pip | 24.0 |
| setuptools | 59.5.0 |
| wheel | 0.42.0 |
| GCC / G++ | 9.5.0 |
| Binutils | 2.39 |
| PyTorch CUDA 构建 | cu111（CUDA 11.1） |

> `cu111` 表示 PyTorch 的 CUDA 构建版本。原始清单未提供 NVIDIA 驱动版本，也未确认独立 CUDA Toolkit 是否已安装。

## 核心依赖

| 类别 | 包名 | 版本 |
| --- | --- | --- |
| 深度学习 | torch | 1.9.1+cu111 |
| 深度学习 | torchvision | 0.10.1+cu111 |
| 深度学习 | torchaudio | 0.9.1 |
| 训练框架 | pytorch-lightning | 1.4.9 |
| 训练指标 | torchmetrics | 0.5.1 |
| 模型库 | timm | 0.4.12 |
| 模型库 | clip | 1.0 |
| 模型库 | open-clip-torch | 2.24.0 |
| 模型扩展 | torch-encoding | 1.2.2b20260929 |
| 数值计算 | numpy | 1.21.6 |
| 科学计算 | scipy | 1.7.3 |
| 数据处理 | pandas | 1.3.5 |
| 图像处理 | Pillow | 9.5.0 |
| 图像处理 | opencv-python | 5.0.0.93 |
| 可视化 | matplotlib | 3.4.3 |
| 实验记录 | tensorboard | 2.11.2 |
| 实验记录 | tensorboardX | 2.6.2.2 |
| 实验记录 | wandb | 0.18.7 |
| 交互界面 | streamlit | 1.22.0 |

## 完整 Python 包清单

以下采用 `包名==版本` 格式记录。包名大小写与部分分隔符统一整理；`thop` 使用提供的 pip 清单版本 `0.1.1.post2209072238`。

<details>
<summary>展开查看完整 Python 依赖</summary>

```text
absl-py==2.1.0
aiohttp==3.8.6
aiosignal==1.3.1
altair==4.2.2
async-timeout==4.0.3
asynctest==0.13.0
attrs==24.2.0
backcall==0.2.0
backports.zoneinfo==0.2.1
backports.functools-lru-cache==2.0.0
blinker==1.6.3
cachetools==5.5.2
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.1.8
clip==1.0
cycler==0.11.0
decorator==5.1.1
docker-pycreds==0.4.0
entrypoints==0.4
filelock==3.12.2
frozenlist==1.3.3
fsspec==2023.1.0
ftfy==6.1.1
future==1.0.0
fvcore==0.1.5.post20221221
gitdb==4.0.12
gitpython==3.1.62
google-auth==2.45.0
google-auth-oauthlib==0.4.6
grpcio==1.62.3
huggingface-hub==0.16.4
idna==3.10
imageio==2.31.2
importlib-metadata==6.7.0
importlib-resources==5.12.0
iopath==0.1.10
ipython==7.33.0
jedi==0.19.1
jinja2==3.1.6
jsonschema==4.17.3
kiwisolver==1.4.5
markdown==3.4.4
markdown-it-py==2.2.0
markupsafe==2.1.5
matplotlib==3.4.3
matplotlib-inline==0.1.7
mdurl==0.1.2
multidict==6.0.5
nose==1.3.7
numpy==1.21.6
oauthlib==3.2.2
open-clip-torch==2.24.0
opencv-python==5.0.0.93
packaging==24.0
pandas==1.3.5
parso==0.8.4
pexpect==4.9.0
pickleshare==0.7.5
pillow==9.5.0
pip==24.0
pkgutil_resolve_name==1.3.10
platformdirs==4.0.0
portalocker==2.7.0
prompt_toolkit==3.0.48
protobuf==3.20.3
psutil==7.2.2
ptyprocess==0.7.0
pyarrow==12.0.1
pyasn1==0.5.1
pyasn1-modules==0.3.0
pydeck==0.8.1b1
pydeprecate==0.3.1
pygments==2.17.2
pympler==1.1
pyparsing==3.1.4
pyrsistent==0.19.3
python-dateutil==2.9.0.post0
pytorch-lightning==1.4.9
pytz==2026.4
pyyaml==6.0.1
regex==2024.4.16
requests==2.31.0
requests-oauthlib==2.0.0
rich==13.8.1
rsa==4.9.1
scipy==1.7.3
sentencepiece==0.2.0
sentry-sdk==2.71.0
setproctitle==1.3.3
setuptools==59.5.0
six==1.17.0
smmap==5.0.3
streamlit==1.22.0
tabulate==0.9.0
tenacity==8.2.3
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboardx==2.6.2.2
termcolor==2.3.0
test_tube==0.7.5
thop==0.1.1.post2209072238
timm==0.4.12
toml==0.10.2
toolz==0.12.1
torch==1.9.1+cu111
torch-encoding==1.2.2b20260929
torchaudio==0.9.1
torchinfo==1.8.0
torchmetrics==0.5.1
torchvision==0.10.1+cu111
tornado==6.2
tqdm==4.68.2
traitlets==5.9.0
typing_extensions==4.7.1
tzlocal==5.1
urllib3==2.0.7
validators==0.20.0
wandb==0.18.7
watchdog==3.0.0
wcwidth==0.2.10
werkzeug==2.2.3
wheel==0.42.0
yacs==0.1.8
yarl==1.9.4
zipp==3.15.0
```

</details>

## 完整 Conda 环境记录

以下保留原始清单的包名、版本、构建及软件源，包括系统工具链和 Python 包。

<details>
<summary>展开查看 Conda 原始清单</summary>

```text
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
absl-py                   2.1.0                    pypi_0    pypi
aiohttp                   3.8.6                    pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
altair                    4.2.2                    pypi_0    pypi
async-timeout             4.0.3                    pypi_0    pypi
asynctest                 0.13.0                   pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                pyhd8ed1ab_4    conda-forge
backports-zoneinfo        0.2.1                    pypi_0    pypi
backports.functools_lru_cache 2.0.0              pyhd8ed1ab_0    conda-forge
binutils_impl_linux-64    2.39                 he00db2b_1    conda-forge
binutils_linux-64         2.39                h5fc0e48_13    conda-forge
blinker                   1.6.3                    pypi_0    pypi
ca-certificates           2025.1.31            hbcca054_0    conda-forge
cachetools                5.5.2                    pypi_0    pypi
certifi                   2026.7.22                pypi_0    pypi
charset-normalizer        3.5.1                    pypi_0    pypi
click                     8.1.8                    pypi_0    pypi
clip                      1.0                      pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
docker-pycreds            0.4.0                    pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
filelock                  3.12.2                   pypi_0    pypi
frozenlist                1.3.3                    pypi_0    pypi
fsspec                    2023.1.0                 pypi_0    pypi
ftfy                      6.1.1                    pypi_0    pypi
future                    1.0.0                    pypi_0    pypi
fvcore                    0.1.5.post20221221          pypi_0    pypi
gcc_impl_linux-64         9.5.0               h99780fb_19    conda-forge
gcc_linux-64              9.5.0               h4258300_13    conda-forge
gitdb                     4.0.12                   pypi_0    pypi
gitpython                 3.1.62                   pypi_0    pypi
google-auth               2.45.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
grpcio                    1.62.3                   pypi_0    pypi
gxx_impl_linux-64         9.5.0               h99780fb_19    conda-forge
gxx_linux-64              9.5.0               h43f449f_13    conda-forge
huggingface-hub           0.16.4                   pypi_0    pypi
idna                      3.10                     pypi_0    pypi
imageio                   2.31.2                   pypi_0    pypi
importlib-metadata        6.7.0                    pypi_0    pypi
importlib-resources       5.12.0                   pypi_0    pypi
iopath                    0.1.10                   pypi_0    pypi
ipython                   7.33.0           py37h89c1867_0    conda-forge
jedi                      0.19.1             pyhd8ed1ab_0    conda-forge
jinja2                    3.1.6                    pypi_0    pypi
jsonschema                4.17.3                   pypi_0    pypi
kernel-headers_linux-64   3.10.0              he073ed8_18    conda-forge
kiwisolver                1.4.5                    pypi_0    pypi
ld_impl_linux-64          2.39                 hcc3a1bd_1    conda-forge
libffi                    3.4.2                h7f98852_5    conda-forge
libgcc                    14.2.0               h77fa898_1    conda-forge
libgcc-devel_linux-64     9.5.0               h0a57e50_19    conda-forge
libgcc-ng                 14.2.0               h69a702a_1    conda-forge
libgomp                   14.2.0               h77fa898_1    conda-forge
libnsl                    2.0.1                hd590300_0    conda-forge
libsanitizer              9.5.0               h2f262e1_19    conda-forge
libsqlite                 3.46.0               hde9e2c9_0    conda-forge
libstdcxx                 14.2.0               hc0a3c3a_1    conda-forge
libstdcxx-devel_linux-64  9.5.0               h0a57e50_19    conda-forge
libstdcxx-ng              14.2.0               h4852527_1    conda-forge
libxcrypt                 4.4.36               hd590300_1    conda-forge
libzlib                   1.3.1                h4ab18f5_1    conda-forge
markdown                  3.4.4                    pypi_0    pypi
markdown-it-py            2.2.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.4.3                    pypi_0    pypi
matplotlib-inline         0.1.7              pyhd8ed1ab_0    conda-forge
mdurl                     0.1.2                    pypi_0    pypi
multidict                 6.0.5                    pypi_0    pypi
ncurses                   6.5                  h59595ed_0    conda-forge
nose                      1.3.7                    pypi_0    pypi
numpy                     1.21.6                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
open-clip-torch           2.24.0                   pypi_0    pypi
opencv-python             5.0.0.93                 pypi_0    pypi
openssl                   3.3.1                h4ab18f5_1    conda-forge
packaging                 24.0                     pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
parso                     0.8.4              pyhd8ed1ab_0    conda-forge
pexpect                   4.9.0              pyhd8ed1ab_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.5.0                    pypi_0    pypi
pip                       24.0               pyhd8ed1ab_0    conda-forge
pkgutil-resolve-name      1.3.10                   pypi_0    pypi
platformdirs              4.0.0                    pypi_0    pypi
portalocker               2.7.0                    pypi_0    pypi
prompt-toolkit            3.0.48             pyha770c72_0    conda-forge
protobuf                  3.20.3                   pypi_0    pypi
psutil                    7.2.2                    pypi_0    pypi
ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
pyarrow                   12.0.1                   pypi_0    pypi
pyasn1                    0.5.1                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pydeck                    0.8.1b1                  pypi_0    pypi
pydeprecate               0.3.1                    pypi_0    pypi
pygments                  2.17.2             pyhd8ed1ab_0    conda-forge
pympler                   1.1                      pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
pyrsistent                0.19.3                   pypi_0    pypi
python                    3.7.12          hf930737_100_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
python_abi                3.7                     4_cp37m    conda-forge
pytorch-lightning         1.4.9                    pypi_0    pypi
pytz                      2026.4                   pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
readline                  8.2                  h8c095d6_2    conda-forge
regex                     2024.4.16                pypi_0    pypi
requests                  2.31.0                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rich                      13.8.1                   pypi_0    pypi
rsa                       4.9.1                    pypi_0    pypi
scipy                     1.7.3                    pypi_0    pypi
sentencepiece             0.2.0                    pypi_0    pypi
sentry-sdk                2.71.0                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                59.5.0                   pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
smmap                     5.0.3                    pypi_0    pypi
sqlite                    3.46.0               h6d4b2fc_0    conda-forge
streamlit                 1.22.0                   pypi_0    pypi
sysroot_linux-64          2.17                h0157908_18    conda-forge
tabulate                  0.9.0                    pypi_0    pypi
tenacity                  8.2.3                    pypi_0    pypi
tensorboard               2.11.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorboardx              2.6.2.2                  pypi_0    pypi
termcolor                 2.3.0                    pypi_0    pypi
test-tube                 0.7.5                    pypi_0    pypi
thop                      0.1.1-2209072238          pypi_0    pypi
timm                      0.4.12                   pypi_0    pypi
tk                        8.6.13          noxft_h4845f30_101    conda-forge
toml                      0.10.2                   pypi_0    pypi
toolz                     0.12.1                   pypi_0    pypi
torch                     1.9.1+cu111              pypi_0    pypi
torch-encoding            1.2.2b20260929           pypi_0    pypi
torchaudio                0.9.1                    pypi_0    pypi
torchinfo                 1.8.0                    pypi_0    pypi
torchmetrics              0.5.1                    pypi_0    pypi
torchvision               0.10.1+cu111             pypi_0    pypi
tornado                   6.2                      pypi_0    pypi
tqdm                      4.68.2                   pypi_0    pypi
traitlets                 5.9.0              pyhd8ed1ab_0    conda-forge
typing-extensions         4.7.1                    pypi_0    pypi
tzdata                    2026c                h151e31d_0    conda-forge
tzlocal                   5.1                      pypi_0    pypi
urllib3                   2.0.7                    pypi_0    pypi
validators                0.20.0                   pypi_0    pypi
wandb                     0.18.7                   pypi_0    pypi
watchdog                  3.0.0                    pypi_0    pypi
wcwidth                   0.2.10             pyhd8ed1ab_0    conda-forge
werkzeug                  2.2.3                    pypi_0    pypi
wheel                     0.42.0             pyhd8ed1ab_0    conda-forge
xz                        5.2.6                h166bdaf_0    conda-forge
yacs                      0.1.8                    pypi_0    pypi
yarl                      1.9.4                    pypi_0    pypi
zipp                      3.15.0                   pypi_0    pypi
```

</details>

## 环境复现说明

当前清单用于记录参考环境。完整安装流程尚待验证，后续需补充：

- Conda 环境导出文件及环境创建步骤。
- PyTorch CUDA 11.1 构建的安装来源。
- `clip`、`torch-encoding` 等包的具体安装来源或源码提交。
- GPU 型号、NVIDIA 驱动及 CUDA Toolkit 信息（如需编译 CUDA 扩展）。

`pip list` 中的版本记录不能完整表达包的安装来源、编译选项及系统依赖，因此暂不将上述清单作为已验证的安装文件。

## 环境检查

在参考环境中，可通过以下命令检查 Python、核心库及 GPU 可用性：

```bash
python --version
python -m pip check

python - <<'PYTHON'
import numpy as np
import torch
import torchvision

print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYTHON
```

以上命令为检查方法，本文暂未记录其执行结果。
