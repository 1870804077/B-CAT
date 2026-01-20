定义huggingface全局变量值

export HF_ENDPOINT=https://hf-mirror.com

git 开发指南
# 查看修改内容
git status
# 添加修改的文件（. 表示所有修改，也可指定具体文件）
git add .
# 提交修改（备注清晰，说明做了什么）
git commit -m "feat: 新增XX功能 / fix: 修复XXbug"
# 1. 拉取上游仓库的最新代码（针对 Fork 场景）
git fetch upstream
# 2. 切换到本地主分支，同步上游主分支
git checkout main
git merge upstream/main
# 3. 切回自己的开发分支，合并最新的主分支代码
git checkout huanse-branch
git merge main
git push origin huanse-branch