"""
tpu_train.py
纯净版 TPU 强化学习训练引擎 (带有 TensorBoard 实时监控)
"""
import os
import time
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from torch.utils.tensorboard import SummaryWriter # 极轻量级引入用于画图

from tpu_model import GoTransformerTPU, train_step
from data_utils import TPUSelfPlayDataset, create_dataloader
from config import ModelConfig, BayesianConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_train_state(rng, config: ModelConfig, learning_rate: float):
    model = GoTransformerTPU(config)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)['params']
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def main():
    parser = argparse.ArgumentParser(description='TPU 极速强化学习训练')
    parser.add_argument('--data-dir', type=str, default='./tpu_data', help='自对弈数据目录')
    parser.add_argument('--checkpoint-dir', type=str, default='./tpu_checkpoints', help='模型保存路径')
    parser.add_argument('--log-dir', type=str, default='./tpu_logs', help='TensorBoard日志路径')
    parser.add_argument('--batch-size', type=int, default=512, help='TPU 批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--disable-bayesian', action='store_true', help='关闭贝叶斯优化')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 启动 TPU 训练引擎 (带 TensorBoard)")
    print(f"当前可用 TPU/GPU 设备: {jax.devices()}")
    print("=" * 60)

    use_bayesian = not args.disable_bayesian
    config = ModelConfig(
        d_model=256, nhead=8, num_layers=8, use_bayesian=use_bayesian
    )
    if use_bayesian:
        print("💡 [核心特性] 贝叶斯不确定性网络头已开启")

    # 初始化 TensorBoard Writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"📊 TensorBoard 日志将保存在: {args.log_dir}")
    print("   (在终端运行: tensorboard --logdir=./tpu_logs 查看曲线)")

    # 数据流水线
    dataset = TPUSelfPlayDataset(data_dir=args.data_dir)
    if len(dataset) == 0:
        print("❌ 未找到数据。请先运行 tpu_selfplay.py 生成数据！")
        return
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 编译环境与模型
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config, args.lr)
    jitted_train_step = jax.jit(train_step, static_argnums=(2,))

    # Orbax 检查点管理器
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(os.path.abspath(args.checkpoint_dir), options=options)

    # 训练循环
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch_idx, batch_pt in enumerate(dataloader):
            obs_pt, policy_pt, value_pt = batch_pt
            value_int = value_pt.long()

            batch_jax = (
                jnp.array(obs_pt.numpy()),
                jnp.array(policy_pt.numpy()),
                jnp.array(value_int.numpy(), dtype=jnp.int32)
            )

            state, metrics = jitted_train_step(state, batch_jax, config)

            epoch_loss += metrics['total_loss'].item()
            global_step += 1

            # 记录到 TensorBoard
            writer.add_scalar('Loss/Total', metrics['total_loss'].item(), global_step)
            writer.add_scalar('Loss/Policy', metrics['policy_loss'].item(), global_step)
            writer.add_scalar('Loss/Value', metrics['value_loss'].item(), global_step)
            if use_bayesian:
                writer.add_scalar('Loss/Uncertainty', metrics['uncertainty_loss'].item(), global_step)

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {global_step} | Loss: {metrics['total_loss']:.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"✅ Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f} | 耗时: {time.time()-start_time:.1f}s")

        checkpoint_manager.save(global_step, args=ocp.args.StandardSave(state))

        checkpoint_manager.wait_until_finished()

        writer.close()
        print("🎉 训练圆满完成！日志已写入 TensorBoard。")

if __name__ == "__main__":
    main()