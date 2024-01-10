
import time
import random

import cv2
from sb3_contrib import MaskablePPO

from mouse_game_wrapper_mlp import MouseEnv

MODEL_PATH = r"trained_models_mlp/ppo_mouse_final"

NUM_EPISODE = 100

RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

board_size = 6
if RENDER:
    env = MouseEnv(seed=seed, board_size=board_size, limit_step=False, silent_mode=False)
else:
    env = MouseEnv(seed=seed, board_size=board_size, limit_step=False, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

sucCount = 0
for episode in range(NUM_EPISODE):
    print(f"=================== Episode {episode + 1} ==================")
    obs = env.reset()
    env.game.render()
    # env.game.showInfo("-" * 30 + "first:")

    done = False
    num_step = 0
    info = None

    while not done:
        # action, _ = model.predict(obs, action_masks=env.get_action_mask())
        action, _ = model.predict(obs)

        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction

        num_step += 1
        obs, reward, done, info = env.step(action)
        env.game.render()

        if done:
            last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            env.game.showInfo("-" * 30 + f"steps:{num_step} end:")
            print("last_action:", last_action)
            rltValue = info["result"]
            if rltValue == 3: # mouse obtained cheese
                sucCount += 1
        else:
            # print(f"{num_step}:", env.game.mousePos)
            pass

    # 场景中止
    env.game.finalize(episode)

sucRatio = "%.3f" % (1.0 * sucCount / NUM_EPISODE)
print(f"suc={sucCount} fail={NUM_EPISODE-sucCount} sucRatio={sucRatio}")
env.close()