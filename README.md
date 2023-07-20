# Deep Reinforcement Learning Crypto Trade

### 🔥 Reference

[Medium Blog by Harsha](https://medium.com/coinmonks/deep-reinforcement-learning-for-trading-cryptocurrencies-5b5502b1ece1)

### 👍 Introduction

This project is to build an intelligent agent to automate trading in crypto markets. 

In this project, deep reinforcement learning algorithms, specifically, **Duelling Deep Q Network** was used to implement the system.

I also tried to publish this project in a flask web app. Modify app.py and vercel.json files to update your configuration.

### 🔢 Methodology

In file DRL_Env, it defines the cryptocurrencies market, and the agent will perform the sell and buy actions in this environment. The environment observes
the action, and either rewards or punishes the agent, based on its action through the Reward mechanism. The agent will collect the reward and again takes
an action by observing the next state of the environment.

#### State

An Observation space is a matrix/vector representation of the state that the agent observes. It can be considered as the input to our agent. It allows us to engineer the features and pass them as input.

The feature space considered in the analysis is a vector representation of:

- N days of asset returns, volume changes and volatility
- Sector ETF returns and volatility (Bitcoin in our case)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Indicator (RSI)
- Bollinger Bands
- The proportion of capital after each trade
- Previous action that was taken by the Agent on the previous state

#### Action Space

The Action space denotes the range of actions our agent can choose from given a state representation. They are:

- -1 = Sell the asset
- 0 = Do Nothing
- 1 = Buy the asset

The agent communicates either to Buy or Sell by choosing a value {-1, 0, 1} from the above-given definition. The mechanism by which we will decode the action values depends on the agent’s underlying algorithm.

#### Reward

The reward **R(t)**, which will be received by the agent after taking action **A(t)** on observing state **S(t)**, can be computed as:

```
R(t) = r(t) * A(t) - |(A(t) - A(t-1))| * C
```

Where:
- **r(t)** is the future return for the asset at time t.
- **A(t)** is the agent's action at time t, taking values from the set {-1, 0, 1}.
- **S(t)** is the vector representation of the state at time t.
- **C** represents transaction costs, assumed to be in the range of 1-5 basis points per trade. (1 basis point = 0.0001 or 0.01%)
