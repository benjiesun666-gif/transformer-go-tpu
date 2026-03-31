"""
tpu_selfplay.py - 自动加载权重 + 贝叶斯热启动版
"""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import orbax.checkpoint as ocp

from config import ModelConfig
from tpu_model import GoTransformerTPU
from pgx_mctx_bridge import PgxMctxMCTS

# --- 实验配置 ---
BATCH_SIZE = 64        # 建议正式测试用 128
NUM_SIMULATIONS = 800    # 搜索深度
MAX_MOVES = 200         # 19x19 建议至少 300 手
CHECKPOINT_DIR = "./tpu_checkpoints"

# --- 💡 你的洞察：贝叶斯开关 ---
USE_BAYESIAN_IN_SELFPLAY = False # 早期建议设为 False，训练几千步后再开


def load_latest_params(init_params):
    if not os.path.exists(CHECKPOINT_DIR):
        return init_params

    # ✅ 使用官方 Manager，它会自动跳过损坏的检查点
    options = ocp.CheckpointManagerOptions(step_prefix='', cleanup_tmp_directories=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(CHECKPOINT_DIR),
        ocp.StandardCheckpointer(),
        options=options
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        print("⚠️ 未发现有效检查点，使用随机参数。")
        return init_params

    print(f"📦 发现有效检查点：Step {latest_step}，正在加载...")
    restored = mngr.restore(latest_step)

    # 兼容处理：有些版本的 restored 直接是 params，有些是包含 params 的字典
    if 'params' in restored:
        return restored['params']
    return restored

def run_selfplay():
    print(f"🚀 初始化 TPU 自我对弈引擎 (Bayesian: {USE_BAYESIAN_IN_SELFPLAY})")
    rng = jax.random.PRNGKey(int(time.time()))

    # 1. 模型与参数加载
    config = ModelConfig()
    model = GoTransformerTPU(config)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, 19, 19, 17), dtype=jnp.float32)

    # 获取随机初始参数
    init_params = model.init(init_rng, dummy_obs)['params']
    # 💡 尝试加载最新权重
    params = load_latest_params(init_params)

    # 2. 桥接与环境
    env = pgx.make("go_19x19")
    # 将贝叶斯开关传给搜索器
    mcts = PgxMctxMCTS(model.apply, num_simulations=NUM_SIMULATIONS, use_bayesian=USE_BAYESIAN_IN_SELFPLAY)

    @jax.jit
    def play_step(params, rng_key, state):
        rng_mcts, rng_action = jax.random.split(rng_key)
        action_weights, _ = mcts.search_batch(params, rng_mcts, state)
        action = jax.random.categorical(rng_action, jnp.log(action_weights + 1e-8))
        next_state = jax.vmap(env.step)(state, action)
        return next_state, action_weights, action

    # 3. 对弈循环
    rng, env_rng = jax.random.split(rng)
    state = jax.vmap(env.init)(jax.random.split(env_rng, BATCH_SIZE))
    trajectories = {"obs": [], "policy": [], "player": [], "mask": []}

    print(f"🔥 对弈开始...")
    start_time = time.time()

    for step in range(MAX_MOVES):
        rng, step_rng = jax.random.split(rng)
        active_mask = ~state.terminated
        current_obs = state.observation

        state, action_weights, action = play_step(params, step_rng, state)

        trajectories["obs"].append(current_obs)
        trajectories["policy"].append(action_weights)
        trajectories["player"].append(state.current_player)
        trajectories["mask"].append(active_mask)

        if (step + 1) % 20 == 0:
            print(f"  > 第 {step+1} 手 | 活跃局数: {np.sum(np.array(active_mask))}")
        if not jnp.any(active_mask): break

    # 4. 数据提取与保存 (逻辑保持你之前的异步优化版)
    print(f"📦 正在异步压制数据并同步...")
    final_rewards = np.array(state.rewards)
    all_obs = np.array(jnp.stack(trajectories["obs"]))
    all_policy = np.array(jnp.stack(trajectories["policy"]))
    all_players = np.array(jnp.stack(trajectories["player"]))
    all_masks = np.array(jnp.stack(trajectories["mask"]))

    seq_len = all_obs.shape[0]
    true_values = np.zeros((seq_len, BATCH_SIZE), dtype=np.float32)
    batch_idx = np.arange(BATCH_SIZE)
    for t in range(seq_len):
        true_values[t] = np.where(all_masks[t], final_rewards[batch_idx, all_players[t]], 0.0)

    true_values_bucket = ((true_values + 1.0) / 2.0 * 127.0).astype(np.int32)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("./tpu_data", exist_ok=True)
    save_path = f"./tpu_data/selfplay_{timestamp}.npz"
    np.savez_compressed(save_path, obs=all_obs, policy=all_policy, value=true_values_bucket, mask=all_masks)

    print(f"🎉 完成！总耗时: {time.time()-start_time:.2f}s | 保存至: {save_path}")

if __name__ == "__main__":
    run_selfplay()