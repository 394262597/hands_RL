# hands_RL
手撕RL的常见算法，不写环境，主要写各个算法的原理。

## DQN
面试常见问题：
- DQN的原理（target net+policy net）
- 目标网络如何做更新？（复制一个policy net，不同时做更新）
- DQN怎么增加的探索？（episilon greedy）
- Q值更新公式？
$$Q(s,a)=Q(s,a)+\alpha [r+\gamma *max Q(s',a')-Q(s,a)]$$
- 经验回放的目的？（打乱顺序，满足独立性假设；提高样本效率）

## Policy gradient
面试常见问题：
- 策略梯度的目标函数？（V的期望）
- V函数的定义？
$$ \sum_{a} \pi_\theta (a|s) Q(s,a) $$
- 