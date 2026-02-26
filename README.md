# PyTorch 面试练习网页（LeetCode 风格）

本项目是一个本地可运行的 PyTorch 练习工具，支持：

- 左侧题目区（题面、难度、标签）
- 左侧 `题目/题解` 切换（题解含讲解 + 参考代码）
- 右上代码区（Monaco 高亮编辑器，失败时自动降级为普通编辑框）
- 右下结果区（测试/提交结果）
- 顶部 `🔧 反馈`：可直接编辑题目描述/起始代码/题解讲解/题解代码并保存
- `测试`：可见用例 + 自定义少量用例
- `提交`：隐藏完整用例
- 题目收藏（本地存储）

## 运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动服务：

```bash
python app.py
```

3. 打开浏览器：

```text
http://127.0.0.1:8765
```

## 题库（当前）

1. 手写 Scaled Dot-Product Attention
2. 手写 Multi-Head Self-Attention.forward
3. 手写 Transformer Encoder Block.forward
4. 手写 CNN LeNet-5.forward
5. 手写 RMSNorm.forward
6. 手写 RoPE 旋转位置编码
7. 手写 LayerNorm.forward
8. 手写 SwiGLU FFN.forward
9. 手写 LoRA Linear.forward
10. 手写单步解码 Attention + KV Cache

说明：

- 判题始终基于后端隐藏用例与参考实现。
- 当你点击 `题解` 时，会按需加载该题官方参考代码与讲解。
- 当你点击 `🔧 反馈` 并保存时，覆盖内容会落盘到 `data/problem_overrides.json`。
- 输入输出构造、对拍和误差比较由平台处理。
