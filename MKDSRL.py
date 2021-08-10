'''
Code for solving Mario Kart DS Figure 8 Circuit (uses DeSmuMe Nintendo DS Emulator)
Control Hotkeys:
X: accelerate
Q: Item
Right arrow key: Turn right
Left arrow key: Turn left
W: Drift
Network is a Convolutional DDQN
It takes 5 contiguous frames of a resized minimap (obtained via screenshot)
and decides between one of six actions (or twelve if drift is set to True):
1. Turn left
2. Turn right
3. Go straight
4. Turn left and use item
5. Turn right and use item
6. Go straight and use item
(for drift, the other six actions go in the same exact order, but the bot also drifts)
Uses one pixel of bottom screen to determine speed and direction (reward)
Bottom screen pixel made by custom Lua Script file that runs in tandem to this code
(more details on Lua Script in the Lua Script file)
One race is one episode (uses more reference pixels to determine when finished)
'''
from tensorflow.keras.layers import Dense, Input, Flatten, Conv3D, MaxPool3D
from pynput.keyboard import Key, Controller, Events
from PIL import ImageGrab
from tensorflow.keras.models import Model
import time
import numpy as np
import random
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Initializing keyboard
keyboard = Controller()
# Setting seed
random.seed(1)
np.random.seed(1)


def get_screen():  # Retrieves screenshot of DS box
    screen = ImageGrab.grab(bbox=(625, 125, 1275, 1125))
    return screen


def is_equal(lis, lis2): #For RGB Value determination
    i = len(lis2)
    if lis[0] == lis2[0]:
        for l in range(i - 1):
            if lis[l] != lis2[l]:
                return False
        return True
    return False


def get_speed_n_dir(pixel):  # Gets the checkpoint for direction checking
    # And speed for other rewards
    if pixel[2] == 100: # Going in right direction
        if pixel[1] == 200:
            speed_dir = 0.75
        elif pixel[1] == 150:
            speed_dir = 0.5
        elif pixel[1] == 100:
            speed_dir = 0.25
        else:
            speed_dir = 0.125
    else:  # Going in wring direction
        if pixel[1] == 200:
            speed_dir = -0.75
        elif pixel[1] == 150:
            speed_dir = -0.5
        elif pixel[1] == 100:
            speed_dir = -0.25
        else:
            speed_dir = -0.125
    return speed_dir


def is_finished(screen):  # Checks reference pixels to make sure episode has ended
    screen = np.array(screen)
    if is_equal(screen[536][316], [69, 69, 158]) and is_equal(screen[531][197], [255, 250, 80]) and \
        is_equal(screen[542][51], [184, 103, 20]):
        return True
    return False


def actt(DQN_output):  # Presses appropriate keys given the action (includes drift)
    inc = DQN_output
    if inc == 0:
        keyboard.release('q')
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        keyboard.release('w')
    elif inc == 1:
        keyboard.press('q')
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        keyboard.release('w')
    elif inc == 2:
        keyboard.release('q')
        keyboard.press(Key.right)
        keyboard.release(Key.left)
        keyboard.release('w')
    elif inc == 3:
        keyboard.press('q')
        keyboard.press(Key.right)
        keyboard.release(Key.left)
        keyboard.release('w')
    elif inc == 4:
        keyboard.release('q')
        keyboard.release(Key.right)
        keyboard.press(Key.left)
        keyboard.release('w')
    elif inc == 5:
        keyboard.press('q')
        keyboard.release(Key.right)
        keyboard.press(Key.left)
        keyboard.release('w')
    elif inc == 6:
        keyboard.release('q')
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        keyboard.press('w')
    elif inc == 7:
        keyboard.press('q')
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        keyboard.press('w')
    elif inc == 8:
        keyboard.release('q')
        keyboard.press(Key.right)
        keyboard.release(Key.left)
        keyboard.press('w')
    elif inc == 9:
        keyboard.press('q')
        keyboard.press(Key.right)
        keyboard.release(Key.left)
        keyboard.press('w')
    elif inc == 10:
        keyboard.release('q')
        keyboard.release(Key.right)
        keyboard.press(Key.left)
        keyboard.press('w')
    else:
        keyboard.press('q')
        keyboard.release(Key.right)
        keyboard.press(Key.left)
        keyboard.press('w')


def get_reward(speed):  # Reward function
    reward = speed * 2 # Reward based on speed
    return reward


class DQN():
    def __init__(self, ddqn, drift=False, episodes=2000, load=False):
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 0.9
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'ddqn_MKDS.h5' if ddqn else 'dqn_MKDS.h5'
        self.n_outputs = 12 if drift else 6
        # Q Network for training
        self.q_model = self.build_model(self.n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = self.build_model(self.n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        self.ddqn = True if ddqn else False
        if self.ddqn: #Loads in weights file if there is one
            print("----------Double DQN--------")
        else:
            print("-------------DQN------------")
        if load: #Can load in weight file to continue training
            if self.ddqn:
                try:
                    self.target_q_model.load_weights('ddqn_MKDS.h5')
                    self.q_model.load_weights('ddqn_MKDS.h5')
                except FileNotFoundError:
                    print("There isn't a file to be loaded")
                try:
                    self.target_q_model.load_weights('ddqn_MKDS.h5')
                    self.q_model.load_weights('ddqn_MKDS.h5')
                except FileNotFoundError:
                    print("There isn't a file to be loaded")
    def build_model(self, n_outputs): #  Network architecture
        inputs = Input(shape=(5, 84, 140, 3), name='state')
        conv = Conv3D(64, (2, 4, 4), activation='relu')(inputs)
        conv = Conv3D(64, (1, 4, 4), activation='relu')(conv)
        conv = MaxPool3D((1, 2, 2))(conv)
        conv = Conv3D(64, (2, 3, 3), activation='relu')(conv)
        conv = Conv3D(64, (1, 3, 3), activation='relu')(conv)
        conv = MaxPool3D((1, 2, 2))(conv)
        conv = Conv3D(64, (2, 2, 2), activation='relu')(conv)
        conv = Conv3D(32, (1, 2, 2), activation='relu')(conv)
        conv = Conv3D(16, (2, 2, 2), activation='relu')(conv)
        x = Flatten()(conv)
        x = Dense(16, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs, activation='linear', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)

    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            rand_action = np.random.choice(6)
            actt(rand_action)
            return rand_action

        # exploit
        q_values = self.target_q_model.predict(state)
        best_action = np.argmax(q_values[0])
        actt(best_action)
        return best_action

    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)
    # compute Q_max
    # use of target Q Network solves the non-stationarity problem

    def forget(self, length):
        for i in range(length):
            self.memory.pop(0)

    def get_target_q_value(self, next_state, reward):
        # max Q value among next state's actions
        if self.ddqn:
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value
            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=True)
        # update exploration-exploitation probability
        self.update_epsilon()
        # copy new params on old target after every 3 training updates
        if self.replay_counter % 5 == 0:
            self.update_weights()

        self.replay_counter += 1
    # decrease the exploration, increase exploitation

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


episode_count = 2000
batch_size = 300
scores = []
running = False
race_length = []
agent = DQN(ddqn=True)
episode_count = 2000
batch_size = 300
scores = []
running = False
race_length = []
for episode in range(episode_count):  # Main training loop
    test_done = False
    total_reward = 0
    while not test_done:
        screen = np.array(get_screen())
        # Doesn't start predicting until some time after black screen shows up
        if is_equal(screen[234][316], [0, 0, 0]) and is_equal(screen[531][197], [0, 0, 0]):
            running = True
        while running:
            time.sleep(5.25)  # So it gets the boost (makes training easier)
            keyboard.press('x')
            time.sleep(2)
            frame = 1
            state = []
            screen = np.array(get_screen())
            same_reward = 0
            while not is_finished(screen):
                screen = get_screen()
                etat = np.array(screen.resize((84, 140))) #One frame
                state.append(etat)
                if frame % 5 == 0 and frame / 5 != 0:
                    state = np.array(state).reshape((1, 5, 84, 140, 3))
                    action = agent.act(state)
                    screen = get_screen()
                    speed = get_speed_n_dir(np.array(screen)[515][24])  # [515][24] is reference pixel
                    if frame == 5: #Does not have enough info
                        pass
                    elif frame == 10:
                        reward = get_reward(speed)
                        max_reward = reward
                        agent.remember(prev_state, prev_action, reward, state, is_finished(screen))
                        total_reward += reward
                    elif frame > 10:
                        reward = get_reward(speed)
                        if reward > max_reward:
                            max_reward = reward
                        agent.remember(prev_state, prev_action, reward, state, is_finished(screen))
                        total_reward += reward
                    prev_state = state
                    prev_action = action
                    state = []
                frame += 1
            running = False
            test_done = True
    keyboard.release('x')
    keyboard.release('q')
    keyboard.release(Key.right)
    keyboard.release(Key.left)
    keyboard.release('w')
    scores.append(total_reward)
    mean_score = np.mean(scores)
    # Displaying some statistics
    print('Episode ' + str(episode + 1) + ':')
    print('Highest reward value attained: ' + str(max_reward))
    print('Total score: ' + str(scores[-1]))
    print('Average score :' + str(mean_score))
    # call experience relay
    item_lis = []
    for i in range(3):  # Forgets last two frames when it has finished but computer didn't pick it up
        item_lis.append(agent.memory[-1])
        agent.memory.pop(-1)
    item_lis = list(item_lis)
    agent.remember(item_lis[-1][0], item_lis[-1][1], item_lis[-1][2], item_lis[-1][3], True)
    race_length.append(len(agent.memory) - sum(race_length))
    if episode > 9: #Only remembering last ten races to help with speed of training
        agent.forget(race_length[0])
        race_length.pop(0)
    print(len(agent.memory))
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)
    agent.save_weights() #Saves after every training so there will be something to go back to. Overwrite is true by default, so there are minimal problems
    for i in range(2):  # Sequence of key presses to select Figure-8 Circuit again
        keyboard.press('x')
        time.sleep(0.5)
        keyboard.release('x')
        time.sleep(0.5)
    time.sleep(3)
    keyboard.press('x')
    time.sleep(0.5)
    keyboard.release('x')
    time.sleep(0.5)
    for i in range(2):
        keyboard.press('x')
        time.sleep(0.1)
        keyboard.release('x')
        time.sleep(0.1)
# Plotting score graphs
episodes = (i for i in range(episode_count))
plt.plot(episodes, scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
#Saving weights
agent.save_weights()

