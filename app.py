from flask import Flask, jsonify
from datetime import date, timedelta

import torch

from DRL_Agent_Memory import ReplayMemory
from DRL_DQN_Agent import DQNAgent, DuellingDQN
from DRL_Env import SingleAssetTradingEnvironment
from DRL_Feature_Generator import DataGetter
from DRL_Global_Params import STATE_SPACE

app = Flask(__name__)

def tradeInEnv():
    today = date.today()
    day_begin = today - timedelta(days=60)

    asset_code = "BTC-USD"
    data = DataGetter(asset_code, start_date=day_begin, end_date=today)
    env = SingleAssetTradingEnvironment(data)

    # Load the online model
    agent = DQNAgent(actor_net=DuellingDQN, memory=ReplayMemory())
    online_checkpoint = torch.load('online.pt')
    agent.actor_online.load_state_dict(online_checkpoint)

    act_dict = {0: -1, 1: 1, 2: 0}
    # sell, do nothing, buy
    state = env.reset()
    done = False
    score_te = 0

    while True:
        actions = agent.act(state)
        action = act_dict[actions]
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(-1, STATE_SPACE)
        state = next_state
        score_te += reward
        if done:
            next_action = act_dict[agent.act(state)]
            break
    return env, data, next_action

def transferAction(act):
    if (act == -1):
        return "sell"
    elif (act == 0):
        return "hold"
    else:
        return "buy"

@app.route("/drl/action")
def getPredictAction():
    _, _, na = tradeInEnv()
    return transferAction(na)

@app.route("/drl/profit")
def getTotalProfit():
    env, _, _ = tradeInEnv()
    return str(env.store['running_capital'][-1] - env.store['running_capital'][0])

@app.route("/drl/actionList")
def getActionList():
    env, data, _ = tradeInEnv()
    timeList = data.timeList
    actionList = env.store['action_store']
    capitalList = env.store['running_capital']

    results = [{'date': t.strftime('%Y-%m-%d') , 'action': transferAction(act), 'capital': round(cap, 2) }
               for t, act, cap in zip(timeList, actionList, capitalList)]

    return jsonify(results)

if __name__ == '__main__':
    app.run()