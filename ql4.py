from player import Player, AI, Random
from game import Game
from table import Table
from ui import UI
import numpy as np
import random
from math import atan2
from bisect import bisect
import matplotlib
import matplotlib.pyplot as plt
import time
import os

matplotlib.rc('axes.formatter', useoffset=False)

DATA_PATH = os.getcwd() + '/data/'

class SARSA(Player):

    features = 52 # 4 + 2 + 45+1
    actions = 3 # 0 null, 1 up, 2 down
    rewards = [-1, .1]

    def __init__(self, **kwargs):
        name = 'w-' + '-'.join(str(x) for kv in kwargs.items() for x in kv)
        super().__init__(name)
        try:
            self.weights = np.load(DATA_PATH + self.name + '.npy')
        except FileNotFoundError:
            self.weights = None
            self.set_weights()
        # assert 'alpha' in kwargs
        assert 'gamma' in kwargs
        assert 'max_episodes' in kwargs
        self.__dict__.update(kwargs)
        self.alpha = 1 - self.episodes*(.99 / self.max_episodes)
        if not self.decay:
            self.set_decay(1)
        self.prev_state = None
        self.prev_feature = None
        self.prev_action = None

    def set_weights(self):
        self.weights = np.zeros((self.actions+1, self.features))

    def save_weights(self):
        np.save(DATA_PATH + self.name + '.npy', self.weights)
        print('saved:\n%s' % (DATA_PATH + self.name + '.npy'))

    @property
    def episodes(self):
        return self.weights[-1][0]

    @property
    def decay(self):
        return self.weights[-1][1]

    def set_decay(self, d):
        self.weights[-1][1] = d

    def increment_episodes(self):
        self.weights[-1][0] += 1
        self.update_alpha()
        # self.update_decay()

    def update_alpha(self):
        """Decay learning rate linearly from 1 to .01 over max_episodes."""
        self.alpha = 1 - self.episodes*(.99 / self.max_episodes)

    def update_decay(self):
        """Decay exploration rate linearly from 1 to .01 over max_episodes."""
        self.weights[0][1] = 1 - self.episodes*(.99 / self.max_episodes)

    def state(self, table):
        ball = table.ball
        unit = table.unit
        paddle = table.paddle
        # ball position, velocity
        result = [ball.x/unit, ball.y/unit, atan2(ball.dy, ball.dx), ball.dt]
        # paddle y position
        result.append(paddle.centery/unit)
        # brick health
        result.extend(table.bricks_health)
        # brick count
        result.append(len(table.bricks))
        return result

    def dynamic_features(self, table):
        # paddle position given action
        still = table.paddle.left/table.unit
        left = max(0, table.paddle.left - table.unit) / table.unit
        right = min(52, table.paddle.left + table.unit) / table.unit
        return [[still], [left], [right]]

    def prev_reward(self, ps, s):
        """Return reward for previous states given next state."""
        # 0 if starting state
        if not ps:
            return 0
        # -1 if ball will hit floor, previous state undetermined
        if s[1] >= 87:
            return -1 if ps[1] < 87 else 0
        # .1 per brick hit
        return (s[-1] - ps[-1]) / 10

    def exploration(self, s, F):
        d = self.decay
        self.update_decay()
        # Prob(greedy) = 1 - d
        #              = 1 - d' + d'/k; k = self.actions
        # d = d * k/(k-1)
        d *= self.actions/(self.actions-1)
        greedy = np.random.random() >= d
        if greedy:
            return np.argmax([np.dot(W[i], np.concatenate((s,F[i])))
                              for i in range(3)])
        return np.random.randint(self.actions)

    def strategy_train(self, table):
        ps = self.prev_state
        pf = self.prev_feature
        pa = self.prev_action
        s = self.state(table)
        F = self.dynamic_features(table)
        pr = self.prev_reward(ps, s)
        a = self.exploration(s, F)
        f = F[a]
        W = self.weights

        # take action
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

        # update new state action reward
        self.prev_state = s
        self.prev_feature = f
        self.prev_action = a

        # check if starting state
        if ps is None:
            return
        # update weights
        pq = np.dot(W[pa], np.concatenate((ps, pf)))
        q = np.dot(W[a], np.concatenate((s,f)))
        W_temp = np.array(W)
        W[pa] += self.alpha*(pr + self.gamma*q - pq) * np.concatenate((ps, pf))
        if np.isnan(W).any():
            print(W_temp)
            print(W)
            print(self.alpha*(pr + self.gamma*q - pq) * np.concatenate((ps, pf)))
            exit()
        print(W)


    def strategy(self, table):
        s = self.state(table)
        F = self.dynamic_features(table)
        W = self.weights
        a = np.argmax([np.dot(W[i], np.concatenate((s,f))) for f in F])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

class SARSAEpsilonGreedyPlayer(SARSA):

    def __init__(self, gamma, epsilon, max_episodes):
        super().__init__(gamma=gamma, epsilon=epsilon,
                         max_episodes=max_episodes)

    def update_decay(self):
        """Decay exploration rate exponentially from 1 to epsilon ** self.max_episodes."""
        self.weights[0][1] *= self.epsilon

class SARSALambda(SARSA):

    def __init__(self, gamma, epsilon, lambda_, max_episodes):
        super().__init__(gamma=gamma, epsilon=epsilon, lambda_=lambda_,
                         max_episodes=max_episodes)

    def set_weights(self):
        # state action reward|eligibility
        self.weights = np.zeros((self.states+1, self.actions, 2))

    @property
    def episodes(self):
        return self.weights[0][0][0]

    @property
    def decay(self):
        return self.weights[0][0][1]

    def set_decay(self, d):
        self.weights[0][0][1] = d

    def increment_episodes(self):
        self.weights[0][0][0] += 1
        self.update_alpha()
        self.update_decay()

    def update_decay(self):
        """Decay exploration rate exponentially from 1 to epsilon ** self.max_episodes."""
        self.weights[0][0][1] *= self.epsilon

    def strategy_train(self, table):
        s = self.state(table)
        r = self.state_to_reward(s)
        W = self.weights

        # check if terminal state
        if r:
            W[s][0][0] = r

        # check if starting state
        if self.prev_state is not None:
            ps = self.prev_state
            pa = self.prev_action
            pr = self.prev_reward

            # previous state reward error
            delta = self.alpha*(pr + self.gamma * max(W[s,:,0]) - W[ps][pa][0])
            # increment eligibility
            W[ps][pa][1] += 1

            # update all state reward by error and elgibility
            W[1:,:,0] += self.alpha * delta * W[1:,:,1]
            # decay eligibility
            W[1:,:,1] *= self.gamma * self.lambda_

        # update new state action reward
        self.prev_state = s
        self.prev_action = a = self.exploration(W[s,:,0])
        self.prev_reward = r

        # take action
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

    def strategy(self, table):
        s = self.state(table)
        W = self.weights
        a = np.argmax(W[s,:,0])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()


class SARSAPlayer(Player):

    def __init__(self, weights, name='sarsa'):
        super().__init__(name)
        self.weights = weights

    def strategy(self, table):
        s = SARSA.state(table)
        W = getattr(self, 'weights')
        a = np.argmax(W[s])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()


class Train(Game):

    def __init__(self, table, player, max_lives, max_sets,
                 stat_sets, display_sets):
        super().__init__(table, player, max_lives)
        self.speed_up_flag = True
        self.max_lives = max_lives
        self.max_sets = max_sets
        self.stat_sets = stat_sets
        self.display_sets = display_sets

        self.name = '-'.join((self.player.name, 'training'))

        self.vars = ['bricks_rem', 'hits', 'lives_rem']

        try:
            self.data = np.load(DATA_PATH + self.name + '-data' + '.npy')
            self.set_num = self.data[-1][0]
            if self.max_sets > self.data.shape[1]:
                d = self.max_sets - self.data.shape[1]
                print('adding %d data positions' % d)
                self.data = np.pad(self.data, ((0,0), (0,d)), mode='constant')
        except FileNotFoundError:
            self.data = np.zeros((len(self.vars)+1, self.max_sets), dtype=int)
            self.set_num = 0

        kwargs = {v: self.data[i] for i,v in enumerate(self.vars)}
        self.__dict__.update(kwargs)
        print('episodes %f' % self.player.episodes)
        print('sets %d' % self.set_num)

    def save(self):
        # save weights
        self.player.save_weights()

        # save data
        self.data[-1][0] = self.set_num
        np.save(DATA_PATH + self.name + '-data' + '.npy', self.data)
        print('saved:\n%s' % (DATA_PATH + self.name + '-data' + '.npy'))

    def display(self):
        self.display_data()
        self.display_stats()

    def display_data(self):
        fig, ax = plt.subplots(nrows=len(self.vars), ncols=1, sharex=True)
        fig.suptitle(self.name + '-data')

        ax[-1].set_xlabel('sets (sum of %d lives)' % self.max_lives)
        for i,v in enumerate(self.vars):
            ax[i].set_ylabel(v)

        x = range(1, self.set_num+1)
        for i,c in enumerate('rgb'):
            ax[i].scatter(x, self.data[i][:self.set_num], marker='o', color=c)

        fig.set_size_inches(16, 12)
        plt.savefig(DATA_PATH + self.name + '-data' + '.png')
        print('saved:\n%s' % (DATA_PATH + self.name + '-data' + '.png'))
        # plt.show()
        plt.close()

    def display_stats(self):
        fig, ax = plt.subplots(nrows=len(self.vars),
                               ncols=len(self.stat_sets),
                               sharex='all', sharey='row')
        fig.suptitle(self.name + '-stats')

        cap = 'Units: sum of %d lives per set' % self.max_lives
        fig.text(.5, .02, cap, fontsize=12, horizontalalignment='center')

        colors = list('rgb')
        for i,v in enumerate(self.vars):
            ax[i][0].set_ylabel(v)
            data_arr = getattr(self, v)
            for j,s in enumerate(self.stat_sets):
                ax[-1][j].set_xlabel('sets (prev %d)' % s)
                x = np.linspace(self.stat_sets[j], self.set_num,
                                num=min(10, self.set_num//self.stat_sets[j]),
                                dtype=int)
                y = [np.mean(data_arr[a-s: a]) for a in x]
                e = [np.std(data_arr[a-s: a]) for a in x]
                ax[i][j].errorbar(x, y, e, capsize=2, linestyle='-',
                                  marker='o', color=colors[i])

        fig.set_size_inches(16, 12)
        plt.savefig(DATA_PATH + self.name + '-stat' + '.png')
        print('saved:\n%s' % (DATA_PATH + self.name + '-stat' + '.png'))
        # plt.show()
        plt.close()

        # check weights
        print(np.matrix.round(self.player.weights, 4))

    def advance_player(self):
        self.player.strategy_train(self.table)

    def reset_scores(self):
        self.player.increment_episodes()
        self.hits[self.set_num] += self.rally
        self.lives -= 1
        self.rally = 0

    def end(self):
        self.bricks_rem[self.set_num] = len(self.table.bricks)
        self.lives_rem[self.set_num] = self.lives
        self.lives = self.max_lives
        self.table.reset()

    def train(self):
        while self.set_num < self.max_sets:
            self.play()
            self.set_num += 1
            if self.set_num in self.display_sets:
                self.save()
                self.display()
                # if self.early_stop():
                #     break
            elif not self.set_num % 100:
                print('sets %d' %self.set_num)
                self.save()
        self.save()

    def early_stop(self):
        if self.set_num == self.max_sets:
            return False
        cont = input('continue training this model? (y/n) ')
        cont = True if cont in ('', 'y', 'Y') else False
        if cont:
            return False
        msg = 'early stop: %d sets for \n%s' % (self.set_num, self.name)
        print(msg)
        return True

    def play_game(self):
        table = Table()
        game = Game(table, self.player, lives=self.max_lives)
        ui = UI(game)
        ui.start()

def main():
    m = 50000
    for g in (1, .5, .9, .1, .999):
        for e in (.9, .5, .1, .999):
            t = time.time()
            table = Table()
            player = SARSAEpsilonGreedyPlayer(gamma=g, epsilon=e, max_episodes=m)
            max_lives = 10
            max_sets = m // 10
            stat_sets = [100, 500]
            display_sets = [1,3,10,30, 100, 500] + list(range(1000, max_sets+1, 1000))

            print('train: gamma=%f, epsilon=%f, max_eps=%d' % (g, e, m))
            T = Train(table, player, max_lives, max_sets, stat_sets,
                      display_sets)
            T.train()
            d = time.time() - t
            min, sec = divmod(d, 60)
            print('time elapsed: %s min %s sec' % (int(min), int(sec)))
            # play = input('play sample game? (y/n) ')
            # if play in ('', 'y', 'Y'):
            #     T.play_game()
            # cont = input('train next model? (y/n) ')
            # if not cont in ('', 'y', 'Y'):
            #     return

if __name__ == '__main__':
    main()
