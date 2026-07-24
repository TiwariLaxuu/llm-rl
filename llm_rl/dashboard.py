"""
Gradio Interactive Dashboard for LLM-RL.
"""

from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from llm_rl.logger import logger

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def create_reward_plot(episode_rewards):
    """Generate matplotlib Figure for episode rewards."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if episode_rewards:
        ax.plot(episode_rewards, label="Episode Reward", color="#4f46e5", alpha=0.6)
        if len(episode_rewards) >= 5:
            # Moving average
            window = min(20, len(episode_rewards))
            moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"MA ({window})", color="#ef4444", linewidth=2)
        ax.set_title("Training Reward Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Episode Data Yet", horizontalalignment="center", verticalalignment="center")
    plt.tight_layout()
    return fig


def launch_dashboard(trainer: Any, port: int = 7860, share: bool = False):
    """
    Launch Gradio Web Interface for LLM-RL monitoring and interaction.
    """
    if not GRADIO_AVAILABLE:
        logger.error("Gradio is not installed. Please run `pip install gradio` to launch dashboard.")
        return None

    monitor = trainer.monitor
    llm = trainer.llm

    with gr.Blocks(title="LLM-RL Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🤖 LLM-RL Dashboard\n"
            "**LLM-Assisted Reinforcement Learning Copilot** | Live Monitoring & Interactive Reasoning"
        )

        with gr.Tabs():
            # Tab 1: Live Monitoring & Stats
            with gr.TabItem("📈 Training Monitor"):
                with gr.Row():
                    with gr.Column(scale=2):
                        plot_output = gr.Plot(label="Live Reward Curve")
                        refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                    with gr.Column(scale=1):
                        stats_markdown = gr.Markdown("### Training Statistics\nInitializing...")

                def update_monitor():
                    stats = monitor.get_context_summary()
                    fig = create_reward_plot(monitor.state.episode_rewards)
                    stats_text = (
                        f"### 📊 Live Metrics\n"
                        f"- **Algorithm**: {stats['algorithm_name']}\n"
                        f"- **Environment**: {stats['env_name']}\n"
                        f"- **Timestep**: {stats['timestep']} / {stats['total_timesteps']}\n"
                        f"- **Total Episodes**: {stats['total_episodes']}\n"
                        f"- **Mean Reward (100 eps)**: `{stats['mean_reward']:.2f}`\n"
                        f"- **Min / Max Reward**: `{stats['min_reward']:.2f}` / `{stats['max_reward']:.2f}`\n"
                        f"- **Reward Trend**: **{stats['reward_trend']}**\n"
                        f"- **Policy Loss**: `{stats['policy_loss']}`\n"
                        f"- **Value Loss**: `{stats['value_loss']}`\n"
                        f"- **Entropy**: `{stats['entropy']}`"
                    )
                    return fig, stats_text

                refresh_btn.click(fn=update_monitor, outputs=[plot_output, stats_markdown])
                demo.load(fn=update_monitor, outputs=[plot_output, stats_markdown])

            # Tab 2: Live Chatbot
            with gr.TabItem("💬 Live Chatbot"):
                chatbot = gr.Chatbot(label="LLM-RL Training Copilot", height=400)
                msg_input = gr.Textbox(placeholder="Ask about training progress, reward drop, failure causes...", label="Your Question")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                def user_chat(user_message, history):
                    if not user_message:
                        return "", history
                    history = history or []
                    reply = trainer.chat(user_message)
                    history.append((user_message, reply))
                    return "", history

                send_btn.click(fn=user_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
                msg_input.submit(fn=user_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
                clear_btn.click(fn=lambda: None, outputs=chatbot)

            # Tab 3: Policy Explainability
            with gr.TabItem("🔍 Policy Explanation"):
                gr.Markdown("### Explain Agent Action under State Observation")
                with gr.Row():
                    obs_input = gr.Textbox(label="Observation Vector / State", value="[-0.03, 0.15, 0.08, 1.42]")
                    act_input = gr.Textbox(label="Action Taken", value="1 (Move Right)")
                    meaning_input = gr.Textbox(label="Action Description", value="Apply right force to cart")
                explain_btn = gr.Button("Explain Decision", variant="primary")
                explanation_output = gr.Markdown("Click 'Explain Decision' to generate analysis.")

                def run_explain(obs, act, meaning):
                    return trainer.explain_action(obs, act, action_meaning=meaning)

                explain_btn.click(fn=run_explain, inputs=[obs_input, act_input, meaning_input], outputs=explanation_output)

            # Tab 4: Reward & Replay Analysis
            with gr.TabItem("🎯 Reward & Replay Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Reward Function Analysis")
                        reward_btn = gr.Button("Analyze Reward Signal", variant="primary")
                        reward_output = gr.Markdown()
                        reward_btn.click(fn=lambda: trainer.analyze_rewards(), outputs=reward_output)

                    with gr.Column():
                        gr.Markdown("### Replay Buffer & Failure Analysis")
                        replay_btn = gr.Button("Analyze Replay Buffer", variant="primary")
                        replay_output = gr.Markdown()
                        replay_btn.click(fn=lambda: trainer.analyze_replay_buffer(), outputs=replay_output)

            # Tab 5: Report Generator
            with gr.TabItem("📝 Training Report"):
                report_btn = gr.Button("Generate Training Report", variant="primary")
                report_output = gr.Markdown("Click to generate full report.")
                report_btn.click(fn=lambda: trainer.generate_report(), outputs=report_output)

    demo.launch(server_name="0.0.0.0", server_port=port, share=share, prevent_thread_lock=True)
    logger.info(f"Dashboard launched at http://localhost:{port}")
    return demo
