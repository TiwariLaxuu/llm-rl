"""
Gradio Interactive Dashboard for LLM-RL with Live Background Monitoring & Dynamic Port Discovery.
"""

from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from llm_rl.utils.helpers import find_free_port
from llm_rl.logger import logger

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def create_reward_plot(episode_rewards):
    """Generate matplotlib Figure for live episode rewards."""
    fig, ax = plt.subplots(figsize=(6, 3))
    if episode_rewards:
        ax.plot(episode_rewards, label="Episode Reward", color="#4f46e5", alpha=0.6)
        if len(episode_rewards) >= 5:
            window = min(20, len(episode_rewards))
            moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"Moving Avg ({window})", color="#ef4444", linewidth=2)
        ax.set_title("Live Reward Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Return")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "Collecting Reward Metrics...", horizontalalignment="center", verticalalignment="center")
    plt.tight_layout()
    return fig


def create_loss_plot(policy_losses, value_losses, entropy_history):
    """Generate matplotlib Figure for live policy loss, value loss, and entropy."""
    fig, ax = plt.subplots(figsize=(6, 3))
    has_data = False
    if policy_losses:
        ax.plot(policy_losses, label="Policy Loss", color="#ef4444", alpha=0.85)
        has_data = True
    if value_losses:
        ax.plot(value_losses, label="Value Loss", color="#3b82f6", alpha=0.85)
        has_data = True
    if entropy_history:
        ax.plot(entropy_history, label="Entropy", color="#10b981", alpha=0.85)
        has_data = True

    if has_data:
        ax.set_title("Live Policy / Value Loss & Entropy", fontsize=11, fontweight="bold")
        ax.set_xlabel("Rollout Step")
        ax.set_ylabel("Loss / Entropy")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "Collecting Loss History...", horizontalalignment="center", verticalalignment="center")
    plt.tight_layout()
    return fig


def format_trajectories_info(trajectories):
    """Format recent recorded trajectories into Markdown table."""
    if not trajectories:
        return "*No trajectory episodes recorded yet. Training rollouts in progress...*"

    lines = [
        "### 📜 Recent Episode Trajectories",
        "| Episode | Steps | Total Reward | Sample State (First Step) | Sample Action |",
        "|---|---|---|---|---|"
    ]
    for idx, t in enumerate(trajectories[-10:], 1):
        steps = t.get("steps", "N/A")
        rew = t.get("total_reward", 0.0)
        states = t.get("states", [])
        actions = t.get("actions", [])
        
        sample_st = str(states[0][:4]) if states and isinstance(states[0], list) else str(states[0]) if states else "N/A"
        sample_act = str(actions[0]) if actions else "N/A"
        rew_fmt = f"{rew:.2f}" if isinstance(rew, (int, float)) else str(rew)
        lines.append(f"| Ep #{idx} | {steps} | {rew_fmt} | `{sample_st}` | `{sample_act}` |")

    return "\n".join(lines)


def launch_dashboard(trainer: Any, port: int = 7860, share: bool = False):
    """
    Launch Gradio Web Interface for background LLM-RL live monitoring.
    Automatically finds an available port if the specified port is in use.
    """
    if not GRADIO_AVAILABLE:
        logger.error("Gradio is not installed. Please run `pip install gradio` to launch dashboard.")
        return None

    monitor = trainer.monitor
    target_port = find_free_port(start_port=port)

    with gr.Blocks(title="LLM-RL Live Dashboard") as demo:
        gr.Markdown(
            "# 🤖 LLM-RL Background Training Dashboard\n"
            "**Real-time RL Training Monitor & Live LLM Reasoning Copilot**"
        )

        # Auto-refresh timer every 2 seconds
        timer = gr.Timer(2.0)

        with gr.Tabs():
            # Tab 1: Live Training Monitoring (Reward Plot, Loss Plot, Trajectories)
            with gr.TabItem("📈 Live Monitor & Plots"):
                with gr.Row():
                    reward_plot = gr.Plot(label="Live Reward Curve")
                    loss_plot = gr.Plot(label="Live Loss & Entropy Curves")

                with gr.Row():
                    with gr.Column(scale=1):
                        stats_markdown = gr.Markdown("### Training Statistics\nInitializing...")
                        refresh_btn = gr.Button("🔄 Manual Refresh", variant="secondary")
                    with gr.Column(scale=2):
                        trajectories_output = gr.Markdown("### Trajectory Information\nCollecting rollouts...")

                def update_live_monitor():
                    stats = monitor.get_context_summary()
                    fig_rew = create_reward_plot(monitor.state.episode_rewards)
                    fig_loss = create_loss_plot(
                        monitor.state.policy_loss_history,
                        monitor.state.value_loss_history,
                        monitor.state.entropy_history
                    )
                    stats_text = (
                        f"### 📊 Live Training Metrics\n"
                        f"- **Algorithm**: `{stats['algorithm_name']}`\n"
                        f"- **Environment**: `{stats['env_name']}`\n"
                        f"- **Timestep**: `{stats['timestep']} / {stats['total_timesteps']}`\n"
                        f"- **Total Episodes**: `{stats['total_episodes']}`\n"
                        f"- **Mean Reward (100 eps)**: `{stats['mean_reward']:.2f}`\n"
                        f"- **Min / Max Reward**: `{stats['min_reward']:.2f}` / `{stats['max_reward']:.2f}`\n"
                        f"- **Reward Trend**: **{stats['reward_trend']}**\n"
                        f"- **Policy Loss**: `{stats['policy_loss']}`\n"
                        f"- **Value Loss**: `{stats['value_loss']}`\n"
                        f"- **Entropy**: `{stats['entropy']}`"
                    )
                    traj_text = format_trajectories_info(monitor.state.trajectories)
                    return fig_rew, fig_loss, stats_text, traj_text

                # Auto-refresh via Timer and Manual Refresh via button
                timer.tick(fn=update_live_monitor, outputs=[reward_plot, loss_plot, stats_markdown, trajectories_output])
                refresh_btn.click(fn=update_live_monitor, outputs=[reward_plot, loss_plot, stats_markdown, trajectories_output])
                demo.load(fn=update_live_monitor, outputs=[reward_plot, loss_plot, stats_markdown, trajectories_output])

            # Tab 2: Live LLM Chatbot
            with gr.TabItem("💬 Live LLM Copilot Chat"):
                gr.Markdown("Ask questions while training is running in the background!")
                chatbot = gr.Chatbot(label="LLM-RL Training Assistant", height=420)
                msg_input = gr.Textbox(placeholder="How is training progressing? Why did reward drop?", label="Your Question")
                with gr.Row():
                    send_btn = gr.Button("Send Question", variant="primary")
                    clear_btn = gr.Button("Clear History")

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

            # Tab 3: Policy Action Explanation
            with gr.TabItem("🔍 Policy Explanation"):
                gr.Markdown("### Explain Agent Action under Given State Observation")
                with gr.Row():
                    obs_input = gr.Textbox(label="Observation Vector / State", value="[-0.03, 0.15, 0.08, 1.42]")
                    act_input = gr.Textbox(label="Action Code", value="1")
                    meaning_input = gr.Textbox(label="Action Description", value="Push Cart Right")
                explain_btn = gr.Button("Explain Policy Decision", variant="primary")
                explanation_output = gr.Markdown("Click button to generate policy explanation.")

                def run_explain(obs, act, meaning):
                    return trainer.explain_action(obs, act, action_meaning=meaning)

                explain_btn.click(fn=run_explain, inputs=[obs_input, act_input, meaning_input], outputs=explanation_output)

            # Tab 4: Reward & Replay Analysis
            with gr.TabItem("🎯 Reward & Trajectory Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Reward Function Analysis")
                        reward_btn = gr.Button("Analyze Reward Signals", variant="primary")
                        reward_output = gr.Markdown()
                        reward_btn.click(fn=lambda: trainer.analyze_rewards(), outputs=reward_output)

                    with gr.Column():
                        gr.Markdown("### Replay Buffer & Failure Analysis")
                        replay_btn = gr.Button("Analyze Trajectories", variant="primary")
                        replay_output = gr.Markdown()
                        replay_btn.click(fn=lambda: trainer.analyze_replay_buffer(), outputs=replay_output)

            # Tab 5: Training Report Generator
            with gr.TabItem("📝 Live Training Report"):
                report_btn = gr.Button("Generate Markdown Report", variant="primary")
                report_output = gr.Markdown("Click to generate comprehensive markdown report.")
                report_btn.click(fn=lambda: trainer.generate_report(), outputs=report_output)

    app, local_url, share_url = demo.launch(
        server_name="127.0.0.1",
        server_port=target_port,
        share=share,
        prevent_thread_lock=True,
        theme=gr.themes.Soft()
    )
    logger.info(f"Live Dashboard running at {local_url}")
    return local_url
