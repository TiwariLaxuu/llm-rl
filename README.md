# LLM-RL

**LLM-RL** is an open-source framework for integrating **Large Language Models (LLMs)** into the **reinforcement learning training loop**. Instead of using an LLM only before or after training, LLM-RL enables the model to act as an intelligent training assistant that observes, analyzes, explains, and improves the RL optimization process.

The framework is designed to be lightweight, modular, and compatible with existing reinforcement learning libraries such as **Stable-Baselines3** and **Gymnasium**.

---

## 🚀 Features

- 🤖 LLM-assisted reinforcement learning
- 📈 Live training monitoring
- 💬 Interactive chatbot during training
- 📝 Automatic training report generation
- 🔍 Policy explainability
- 🎯 Reward analysis and suggestions
- 📊 Replay buffer analysis
- ⚡ Async LLM inference
- 🖥️ Gradio dashboard
- 📦 Plug-and-play integration with Stable-Baselines3

---

## Motivation

Current reinforcement learning algorithms rely almost entirely on handcrafted rewards and sampled experience. Debugging failures, understanding agent behavior, and diagnosing unstable training often require significant manual effort.

LLM-RL introduces an intelligent assistant that continuously observes the training process and provides semantic feedback, explanations, and recommendations while the agent is learning.

The LLM **does not replace the policy network**. Instead, it acts as an auxiliary reasoning module that supports reinforcement learning throughout optimization.

---

## Architecture

```text
                 ┌──────────────────────────┐
                 │     PPO / DQN / SAC      │
                 └────────────┬─────────────┘
                              │
                      Stable-Baselines3
                              │
                       Training Callback
                              │
          ┌───────────────────┴────────────────────┐
          │                                        │
          ▼                                        ▼
   Training Monitor                       Async LLM Worker
          │                                        ▲
          │                                        │
          ▼                                        │
 Shared Training State  <──────────────────────────┘
          │
          ├── Timesteps
          ├── Rewards
          ├── Loss
          ├── Entropy
          ├── Observations
          ├── Trajectories
          └── Recommendations
```

---

## Supported Features

### Live Chat

Ask questions while training is running.

Example:

```text
How is training progressing?

Why did reward decrease?

Show difficult trajectories.

What caused the last failure?

Should exploration be increased?

Generate a summary of today's training.
```

---

### Reward Analysis

The LLM analyzes recent trajectories and suggests improvements for sparse or noisy reward functions.

---

### Replay Buffer Analysis

The framework identifies informative transitions, failure cases, and unusual experiences that may deserve additional attention.

---

### Explainability

Generate natural language explanations for agent actions.

Example:

```text
Observation:
[-0.03, 0.15, 0.08, 1.42]

Action:
Move Right

Explanation:
The pole is leaning right with increasing angular velocity.
Moving right helps position the cart beneath the pole to restore balance.
```

---

### Automatic Reports

Generate reports after training containing:

- Training summary
- Reward statistics
- Failure analysis
- LLM recommendations
- Hyperparameter suggestions
- Future improvements

---

## Installation

```bash
pip install llm-rl
```

---

## Quick Start

```python
import gymnasium as gym
from stable_baselines3 import PPO
from llm_rl import LLMTrainer

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env)

trainer = LLMTrainer(
    model=model,
    llm_backend="ollama",
    model_name="llama3.1:8b"
)

trainer.train(100000)
```

---

## Dashboard

Launch the interactive dashboard.

```python
trainer.dashboard()
```

Features include:

- Live reward curve
- Training statistics
- Interactive LLM chat
- Policy explanations
- Episode summaries
- Generated reports

---

## Supported Algorithms

- PPO
- DQN
- A2C
- SAC
- TD3
- Recurrent PPO

---

## Supported LLM Providers

- Ollama
- llamacpp (local model support)
- OpenAI
- Anthropic
- Hugging Face
- Groq
- Gemini

---

---

## Project Structure

```text
llm_rl/
│
├── trainer.py
├── callback.py
├── monitor.py
├── report.py
├── prompts.py
├── dashboard.py
├── explain.py
├── replay.py
├── reward.py
├── logger.py
│
├── llm/
│   ├── ollama.py
│   ├── openai.py
│   ├── huggingface.py
│   └── base.py
│
├── algorithms/
│   ├── ppo.py
│   ├── dqn.py
│   └── wrappers.py
│
└── utils/
```

---

## Research Vision

This project explores a new direction in reinforcement learning where large language models act as **training-time collaborators** rather than replacing the policy network.

Research questions include:

- Can LLM-guided replay analysis improve sample efficiency?
- Can language models assist with reward design?
- Can LLMs explain policy failures during optimization?
- Can semantic reasoning improve RL debugging?
- Can LLMs generate actionable training recommendations?

---

## Contributing

Contributions are welcome.

You can contribute by:

- Adding new RL algorithms
- Supporting additional LLM providers
- Improving dashboards
- Building evaluation benchmarks
- Enhancing explainability modules
- Writing documentation

---

## License

This project is licensed under the **Apache License 2.0**.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_rl,
  title = {LLM-RL: A Framework for LLM-Assisted Reinforcement Learning},
  author = {Laxmi Tiwari},
  year = {2026},
  url = {https://github.com/TiwariLaxuu/llm-rl}
}
```
