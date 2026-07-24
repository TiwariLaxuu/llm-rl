"""
LLM-RL Quickstart Example: CartPole-v1 Training with PPO & LLMTrainer
----------------------------------------------------------------------
This script demonstrates how to integrate LLMTrainer into the RL training loop,
run training for 500 timesteps, explain agent actions, evaluate reward signals,
and generate a post-training markdown report.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from llm_rl import LLMTrainer


def main():
    print("=" * 60)
    print("🚀 LLM-RL CartPole-v1 Example (500 Timesteps)")
    print("=" * 60)

    # 1. Create Gymnasium Environment
    env_id = "CartPole-v1"
    print(f"\n1. Creating environment '{env_id}'...")
    env = gym.make(env_id)

    # 2. Instantiate Stable-Baselines3 PPO Model
    print("2. Initializing PPO agent...")
    model = PPO("MlpPolicy", env, verbose=0)

    # 3. Instantiate LLMTrainer
    print("3. Initializing LLMTrainer with Ollama backend...")
    trainer = LLMTrainer(
        model=model,
        llm_backend="ollama",
        model_name="llama3.1:8b"
    )

    # 4. Train Agent for 500 Timesteps
    print("\n4. Starting RL Training for 500 timesteps...")
    trainer.train(total_timesteps=500)
    print("   ✅ Training finished successfully!")

    # 5. Explain Agent Policy Decision
    print("\n5. Explaining Agent Policy Decision...")
    sample_obs = [-0.02, 0.18, 0.05, 0.45]  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    action, _ = model.predict(sample_obs)
    explanation = trainer.explain_action(
        observation=sample_obs,
        action=int(action),
        action_meaning="Push Cart Right" if action == 1 else "Push Cart Left",
        env_info="CartPole-v1 balancing physics"
    )
    print(f"   Observation: {sample_obs}")
    print(f"   Action Taken: {action}")
    print(f"   LLM Explanation:\n{explanation}")

    # 6. Analyze Reward Signals
    print("\n6. Running Reward Analysis...")
    reward_analysis = trainer.analyze_rewards(notes="Initial CartPole stabilization run.")
    print(f"   LLM Reward Analysis:\n{reward_analysis}")

    # 7. Analyze Replay / Trajectories
    print("\n7. Running Trajectory & Experience Analysis...")
    replay_analysis = trainer.analyze_replay_buffer()
    print(f"   LLM Trajectory Analysis:\n{replay_analysis}")

    # 8. Interactive Chat Query
    print("\n8. Querying LLM Training Copilot...")
    chat_response = trainer.chat("How is the training progressing? What should we improve?")
    print(f"   User: 'How is the training progressing? What should we improve?'")
    print(f"   Copilot Response:\n{chat_response}")

    # 9. Generate Post-Training Markdown Report
    report_file = "cartpole_training_report.md"
    print(f"\n9. Generating training report: '{report_file}'...")
    report_markdown = trainer.generate_report(save_path=report_file)
    print(f"   ✅ Report generated and saved to {report_file}!")

    print("\n" + "=" * 60)
    print("🎉 Example completed cleanly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
