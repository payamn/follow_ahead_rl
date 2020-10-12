import wandb
import pickle
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()

is_write_to_pickel = True
plt.title("Reward")
def draw_to_result_std(steps, value, run, title , color):
    std, smoothed = std_batch(value, 128)
    steps = steps[:-128]

    #plt.plot(steps, smoothed, color, alpha=.25)
    smoothed = np.asarray(smoothed)
    std = np.asarray(std)
    plt.fill_between(steps, smoothed-std, smoothed+std, color=color, alpha=0.1)
    plt.plot(steps, smoothed, color,  label=run, linewidth=0.5)


    plt.xlabel("n iteration")
    plt.ylabel("reward")
    plt.legend(loc='upper left')

    # save image
    plt.savefig("data/"+title+'.svg', format='svg', dpi=1200)

def draw_to_result(steps, value, avg_v, run, title , color):
    plt.plot(steps, value, color,  label=run)
    plt.plot(steps, avg_v, color, alpha=.25)

    plt.xlabel("n iteration")
    plt.ylabel("reward")
    plt.legend(loc='upper left')

    # save image
    plt.savefig(title+".png")  # should before show method

def std_batch(scalars, batch_size):
    std_list = []
    mean_list = []
    for i in range (len(scalars)-batch_size):
        std_list.append(np.std(scalars[i: i+batch_size])/2)
        mean_list.append(np.mean(scalars[i: i+batch_size]))

    return std_list, mean_list


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def write_to_pickel(name, smoothed, value, steps):
    with open(name, "wb") as f:
        pickle.dump({"smoothed":smoothed, "value": value, "steps":steps}, f)

def read_from_pickel(name, smoothed, value, steps):
    with open(name, "rb") as f:
        dic = pickle.load(f)
        return dic["smoothed"], dic["value"], dic["steps"]

def read_from_history(name, scan_history, list_reward=None, steps = None, greater_steps=0, less_than_steps=42000, is_write_to_pickel=False, write_name="nothing", plot=False, color="-g"):
    if list_reward is None or steps is None:
        list_reward = []
        steps = []
    for hist in scan_history:
        if hist["_step"] < greater_steps:
            continue
        list_reward.append(hist[name])
        steps.append(hist["_step"])
        if less_than_steps > 0 and hist["_step"] > less_than_steps:
            break
    smoothed = smooth(list_reward, 0.99)
    if is_write_to_pickel:
         write_to_pickel("data/"+write_name+".pkl", smoothed, list_reward, steps)
    if plot:
        draw_to_result_std(steps, list_reward, write_name, name, color=color)
    return list_reward, steps, smoothed
# Project is specified by <entity/project-name>
run_no_cu = api.run("pnikdel/follow_ahead_d4pg_v1_auto/2bxg6e3i")
run_no_cu2 = api.run("pnikdel/follow_ahead_d4pg_v1_auto/38olka88")
run_ppo = api.run("pnikdel/follow_ahead_d4pg_v1_auto/1igw8ekz")
run_planner_best = api.run("pnikdel/follow_ahead_d4pg_v1_auto/1t9gwf11")
run_cmd_vel = api.run("pnikdel/follow_ahead_d4pg_v1_auto/2o0kap3v")
list_reward_no_cu = []
steps_no_cu = []
list_reward_ppo = []
steps_ppo = []
read_from_history("agent/reward", run_planner_best.scan_history(), is_write_to_pickel=is_write_to_pickel, write_name="Planner", plot=True, color="orange")
read_from_history("agent/reward", run_cmd_vel.scan_history(), is_write_to_pickel=is_write_to_pickel, write_name="cmd_vel", plot=True, color="y")
list_reward_no_cu, steps_no_cu, smoothed_no_cu = read_from_history("agent/reward", run_no_cu2.scan_history(), list_reward_no_cu, steps_no_cu, less_than_steps=1700)
list_reward_no_cu, steps_no_cu, smoothed_no_cu = read_from_history("agent/reward", run_no_cu.scan_history(), list_reward_no_cu, steps_no_cu, greater_steps=1700, is_write_to_pickel=is_write_to_pickel, write_name="Planner_No_Curricullam", plot=True, color="r")
list_reward_ppo, steps_ppo, smoothed_ppo = read_from_history("agent/reward", run_ppo.scan_history(), list_reward_ppo, steps_ppo, is_write_to_pickel=is_write_to_pickel, write_name="PPO", plot=True, color="b")


# show
plt.show()
