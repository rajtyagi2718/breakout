from player import Player, AI, Random
from game import Game
from table import Table
from ui import UI
import numpy as np
import random
from bisect import bisect
import matplotlib
import matplotlib.pyplot as plt
import time
import os

matplotlib.rc('axes.formatter', useoffset=False)

DATA_PATH = os.getcwd() + '/data/'

class QPlayer(Player):

    states = 123649 # 1+23*32*14*12
    actions = 3 # 0 null, 1 up, 2 down
    rewards = [-1]
    theta_dist = np.linspace(-.925, .925, 10)

    def __init__(self, **kwargs):
        name = 'q-' + '-'.join(str(x) for kv in kwargs.items() for x in kv)
        super().__init__(name)
        try:
            self.qtable = np.load(DATA_PATH + self.name + '.npy')
        except FileNotFoundError:
            self.qtable = None
            self.set_qtable()
        # assert 'alpha' in kwargs
        assert 'gamma' in kwargs
        assert 'max_episodes' in kwargs
        self.__dict__.update(kwargs)
        self.alpha = 1 - self.episodes*(.99 / self.max_episodes)
        if not self.decay:
            self.set_decay(1)
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

    def set_qtable(self):
        self.qtable = np.zeros((self.states+1, self.actions))

    def save_qtable(self):
        np.save(DATA_PATH + self.name + '.npy', self.qtable)
        print('saved:\n%s' % (DATA_PATH + self.name + '.npy'))

    @property
    def episodes(self):
        return self.qtable[0][0]

    @property
    def decay(self):
        return self.qtable[0][1]

    def set_decay(self, d):
        self.qtable[0][1] = d

    def increment_episodes(self):
        self.qtable[0][0] += 1
        self.update_alpha()
        # self.update_decay()

    def update_alpha(self):
        """Decay learning rate linearly from 1 to .01 over max_episodes."""
        self.alpha = 1 - self.episodes*(.99 / self.max_episodes)

    def update_decay(self):
        """Decay exploration rate linearly from 1 to .01 over max_episodes."""
        self.qtable[0][1] = 1 - self.episodes*(.99 / self.max_episodes)

    # states = 1+23*32*14*12
    # features = y[0,22]+1 x[0,31] p[0,13] t[0,11]
    @classmethod
    def state(cls, table):
        # feature: ball y position
        y = table.ball.y
        y //= 32
        if y > 22:
            return 1
        # feature: ball x position
        x = min(max(0, table.ball.x // 16), 31)
        # feature: paddle x (left) position
        p = table.paddle.left // 32
        # feature: ball velocity
        dx = table.ball.dx
        dy = table.ball.dy
        if dy <= 0:
            if dx < -.2:
                t = 0
            elif dx < .2:
                t = 1
            else:
                t = 2
        else:
            t = 2 + bisect(cls.theta_dist, dx)

        assert 0 <= y < 23 and 0 <= x < 32 and 0 <= p < 14 and 0 <= t < 12, (y, x, p, t)

        return int(12*(14*(32*y + x) + p) + t)

    def state_to_reward(self, s):
        """Return reward for given state."""
        return -int(s == 1)

    def exploration(self, ar):
        d = self.decay
        self.update_decay()
        # Prob(greedy) = 1 - d
        #              = 1 - d' + d'/k; k = self.actions
        # d = d * k/(k-1)
        d *= self.actions/(self.actions-1)
        greedy = np.random.random() >= d
        return np.argmax(ar) if greedy else np.random.randint(self.actions)

    def strategy_train(self, table):
        s = self.state(table)
        r = self.state_to_reward(s)
        Q = self.qtable

        # check if terminal state
        if r:
            Q[s][0] = r

        # check if starting state
        if self.prev_state is not None:
            ps = self.prev_state
            pa = self.prev_action
            pr = self.prev_reward

            # update previous state action reward
            Q[ps][pa] += self.alpha*(pr + self.gamma * max(Q[s]) - Q[ps][pa])

        # update new state action reward
        self.prev_state = s
        self.prev_action = a = self.exploration(Q[s])
        self.prev_reward = r

        # take action
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

    def strategy(self, table):
        s = self.state(table)
        Q = self.qtable
        a = np.argmax(Q[s])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

class QEpsilonGreedyPlayer(QPlayer):

    def __init__(self, gamma, epsilon, max_episodes):
        super().__init__(gamma=gamma, epsilon=epsilon,
                         max_episodes=max_episodes)

    def update_decay(self):
        """Decay exploration rate exponentially from 1 to epsilon ** self.max_episodes."""
        self.qtable[0][1] *= self.epsilon

class QLambda(QPlayer):

    def __init__(self, gamma, epsilon, lambda_, max_episodes):
        super().__init__(gamma=gamma, epsilon=epsilon, lambda_=lambda_,
                         max_episodes=max_episodes)

    def set_qtable(self):
        # state action reward|eligibility
        self.qtable = np.zeros((self.states+1, self.actions, 2))

    @property
    def episodes(self):
        return self.qtable[0][0][0]

    @property
    def decay(self):
        return self.qtable[0][0][1]

    def set_decay(self, d):
        self.qtable[0][0][1] = d

    def increment_episodes(self):
        self.qtable[0][0][0] += 1
        self.update_alpha()
        self.update_decay()

    def update_decay(self):
        """Decay exploration rate exponentially from 1 to epsilon ** self.max_episodes."""
        self.qtable[0][0][1] *= self.epsilon

    def strategy_train(self, table):
        s = self.state(table)
        r = self.state_to_reward(s)
        Q = self.qtable

        # check if terminal state
        if r:
            Q[s][0][0] = r

        # check if starting state
        if self.prev_state is not None:
            ps = self.prev_state
            pa = self.prev_action
            pr = self.prev_reward

            # previous state reward error
            delta = self.alpha*(pr + self.gamma * max(Q[s,:,0]) - Q[ps][pa][0])
            # increment eligibility
            Q[ps][pa][1] += 1

            # update all state reward by error and elgibility
            Q[1:,:,0] += self.alpha * delta * Q[1:,:,1]
            # decay eligibility
            Q[1:,:,1] *= self.gamma * self.lambda_

        # update new state action reward
        self.prev_state = s
        self.prev_action = a = self.exploration(Q[s,:,0])
        self.prev_reward = r

        # take action
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

    def strategy(self, table):
        s = self.state(table)
        Q = self.qtable
        a = np.argmax(Q[s,:,0])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()


class QLPlayer(Player):

    def __init__(self, qtable, name='ql'):
        super().__init__(name)
        self.qtable = qtable

    def strategy(self, table):
        s = QPlayer.state(table)
        Q = getattr(self, 'qtable')
        a = np.argmax(Q[s])
        if a == 1:
            table.move_left()
        elif a == 2:
            table.move_right()

table_name = 'q-gamma-1-epsilon-0.9-max_episodes-250000'
qtable = np.load(table_name + '.npy')
ql = QLPlayer(qtable)


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
        # save qtable
        self.player.save_qtable()

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
    m = 250000
    for g in (1, .5, .9, .1, .999):
        for e in (.9, .5, .1, .999):
            t = time.time()
            table = Table()
            player = QEpsilonGreedyPlayer(gamma=g, epsilon=e, max_episodes=m)
            max_lives = 10
            max_sets = m // 10
            stat_sets = [10, 30, 100, 300, 1000]
            display_sets = [100, 500] + list(range(1000, max_sets+1, 1000))

            print('train: gamma=%f, epsilon=%f, max_eps=%d' % (g, e, m))
            T = Train(table, player, max_lives, max_sets, stat_sets,
                      display_sets)
            # T.train()
            d = time.time() - t
            min, sec = divmod(d, 60)
            print('time elapsed: %s min %s sec' % (int(min), int(sec)))
            play = input('play sample game? (y/n) ')
            if play in ('', 'y', 'Y'):
                T.play_game()
            cont = input('train next model? (y/n) ')
            if not cont in ('', 'y', 'Y'):
                return

if __name__ == '__main__':
    main()
