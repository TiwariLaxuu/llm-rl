"""
Prompt templates for LLM-RL modules.
Contains structured templates for chatbot assistance, policy explanation, reward analysis,
replay buffer analysis, status summaries, and report generation.
"""

SYSTEM_PROMPT = """You are LLM-RL Assistant, an expert AI and Reinforcement Learning copilot.
Your job is to analyze live training metrics, explain agent behavior, evaluate reward functions,
diagnose training instabilities, analyze replay buffer transitions, and provide actionable recommendations.

Always be concise, precise, objective, and clear in your analysis.
When given numerical data (timesteps, rewards, loss, entropy), interpret trends accurately.
"""

LIVE_CHAT_PROMPT_TEMPLATE = """You are observing an active Reinforcement Learning training session.

### Current Training State:
- Timestep: {timestep} / {total_timesteps}
- Total Episodes: {total_episodes}
- Mean Reward (Recent 100 eps): {mean_reward:.2f} (std: {std_reward:.2f}, min: {min_reward:.2f}, max: {max_reward:.2f})
- Reward Trend: {reward_trend}
- Policy Loss: {policy_loss}
- Value Loss: {value_loss}
- Entropy: {entropy}
- Learning Rate: {learning_rate}
- Algorithm: {algorithm_name}
- Environment: {env_name}

### Trajectory Summary:
{trajectory_summary}

### User Question:
{user_query}

Provide a helpful, insightful, and detailed response addressing the user's question based on the training state.
"""

EXPLAIN_ACTION_PROMPT_TEMPLATE = """You are analyzing the decision made by an RL policy network.

### Context:
- Environment: {env_name}
- State Observation: {observation}
- Action Taken: {action}
- Action Meaning/Description: {action_meaning}
- Environment Info / Additional Context: {env_info}

Explain in clear natural language why the agent took this action under the observed state, what goal it likely aims to achieve, and whether this decision aligns with optimal behavior in this domain.
"""

REWARD_ANALYSIS_PROMPT_TEMPLATE = """You are evaluating the reward function and return statistics of a Reinforcement Learning agent.

### Training Metrics:
- Mean Reward: {mean_reward:.2f}
- Min Reward: {min_reward:.2f}
- Max Reward: {max_reward:.2f}
- Reward Standard Deviation: {std_reward:.2f}
- Recent Episode Rewards: {recent_rewards}

### Trajectory / Task Notes:
{trajectory_notes}

Analyze the reward distribution and trajectory patterns:
1. Is the reward signal sparse, noisy, or dense?
2. Are there signs of reward hacking, sub-optimal local optima, or reward vanishing/exploding?
3. Provide 3 concrete recommendations for reward shaping or penalty adjustments to improve training stability and policy performance.
"""

REPLAY_ANALYSIS_PROMPT_TEMPLATE = """You are analyzing replay buffer transitions and training experiences collected by the RL agent.

### Replay Buffer & Experience Summary:
- Total Transitions Analyzed: {num_transitions}
- High-Loss / Failure Case Count: {failure_count}
- Sample Failure States / Trajectories:
{failure_samples}

- Sample High-Reward / Successful Trajectories:
{success_samples}

Analyze these experiences:
1. What patterns differentiate successful episodes from failed episodes?
2. Are there critical edge cases or bottleneck states where the agent frequently fails?
3. What experience sampling or exploration strategy adjustments (e.g. prioritized replay, entropy schedule, curiosity) would help the agent overcome these failure modes?
"""

REPORT_GENERATION_PROMPT_TEMPLATE = """Generate a comprehensive, professional Markdown training report for a Reinforcement Learning run.

### Training Overview:
- Environment: {env_name}
- Algorithm: {algorithm_name}
- Total Timesteps: {timestep} / {total_timesteps}
- Total Episodes: {total_episodes}
- Final Mean Reward: {mean_reward:.2f}
- Peak Reward: {max_reward:.2f}
- Final Policy Loss: {policy_loss}
- Final Value Loss: {value_loss}
- Reward Trend: {reward_trend}

### Key Metrics History:
{metrics_history_summary}

### Failure & Trajectory Analysis:
{failure_analysis_summary}

Create a well-formatted markdown report containing:
1. **Executive Summary**
2. **Training Progression & Metric Analysis** (Reward curves, Loss, Entropy)
3. **Failure Analysis & Bottlenecks**
4. **LLM Insights & Policy Behavior**
5. **Hyperparameter & Algorithmic Recommendations**
6. **Next Steps & Future Improvements**
"""
