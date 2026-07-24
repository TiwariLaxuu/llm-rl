from setuptools import setup, find_packages

setup(
    name="llm-rl",
    version="0.1.0",
    description="LLM-RL: Reinforcement Learning with LLM-Assisted Training Loop",
    author="Laxmi Tiwari",
    author_email="laxmi@logictronix.com.com",
    url="https://github.com/TiwariLaxuu/llm-rl",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.1",
        "stable-baselines3>=2.0.0",
        "numpy>=1.20.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "gradio>=4.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
