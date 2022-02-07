'''frozen_lake_wrappers module.'''

from sys import exit  # pylint: disable=W0622
import gym
import pygame
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


class FrozenLakeWinWrapper(gym.Wrapper):
    """ FrozenLakeWinWrapper class.

        ver: 0.5
        - added goal property.
        Signals if episode ended reaching the goal

        ver: 0.4
        - added render param to constructor.
        So we can control if the window is rendered or not.

        ver: 0.3
        - added sounds and music

        ver: 0.2
        - added hole and wall rewards
        - added is_closed property
    """

    # pylint: disable=too-many-instance-attributes

    _VER = 0.5

    _CELL_W = 100
    _CELL_H = 100

    _AGENT_IMG = os.path.join(package_directory, 'art', 'robot.png')
    _SND_MUSIC = os.path.join(package_directory, 'sound',
                              'zapsplat_music_arcade_012.ogg')
    _SND_HIT = os.path.join(package_directory, 'sound', 'zapsplat_sound_hit_02.wav')
    _SND_FALL = os.path.join(package_directory, 'sound', 'zapsplat_sound_lose_006.wav')
    _SND_GOAL = os.path.join(package_directory, 'sound', 'zapsplat_sound_power_up_retro_001.wav')

    _DEF_BACK_COL = (255, 255, 255)
    _DEF_GOAL_COL = (255, 255, 0)
    _DEF_GRID_COL = (200, 200, 225)
    _DEF_HOME_COL = (0, 128, 0)
    _DEF_HOLE_COL = (0, 0, 255)

    _DEF_REWD_GOAL = 1.0
    _DEF_REWD_HOLE = 0.0
    _DEF_REWD_WALL = 0.0

    def __init__(self, title="FrozenLake", slippery=True, sound=False, render=True):
        self._env = gym.make("FrozenLake-v0", is_slippery=slippery)
        gym.Wrapper.__init__(self, self._env)

        self._prev_state = 0
        self._state = 0
        self._goal = False  # goal reached
        self._done = False  # episode done
        self._render = render
        self._sound = sound
        self._is_slippery = slippery
        self._rewd_goal = self._DEF_REWD_GOAL
        self._rewd_hole = self._DEF_REWD_HOLE
        self._rewd_wall = self._DEF_REWD_WALL
        self._w = self._env.ncol * self._CELL_W
        self._h = self._env.nrow * self._CELL_H

        # init pygame window
        pygame.init()
        if render:
            self._alive = True  # window alive
            self._scr = pygame.display.set_mode((self._w, self._h), pygame.DOUBLEBUF, 32)
            pygame.display.set_caption(title)

            # set images
            self.set_im_agent(self._AGENT_IMG)
            self._im_goal = None

            # set colors
            self._home_col = self._DEF_HOME_COL
            self._hole_col = self._DEF_HOLE_COL
            self._goal_col = self._DEF_GOAL_COL
            self._grid_col = self._DEF_GRID_COL
            self._back_col = self._DEF_BACK_COL

            # init sound
            if self._sound:
                self.init_sound()
        else:
            self._alive = False

    def reset(self, **kwargs):
        """calls environment's reset() method.
        """
        self._state = self._env.reset(**kwargs)
        self._prev_state = self._state
        self._done = False
        self._goal = False
        return self._state

    def step(self, action):
        """calls environment's step(action) method.
        """
        self._prev_state = self._state
        self._state, reward, self._done, info = self._env.step(action)

        # hit a wall
        if self._state == self._prev_state:
            reward = self._rewd_wall
            if self._sound:
                self._snd_hit.play()

        # fall into a hole
        if self._env.desc.flatten()[self._state] == b'H':
            reward = self._rewd_hole
            if self._sound:
                self._snd_fall.play()

        # reach goal
        if self._env.desc.flatten()[self._state] == b'G':
            self._goal = True
            reward = self._rewd_goal
            if self._sound:
                self._snd_goal.play()

        return self._state, reward, self._done, info

    def set_im_agent(self, im_agent_file):
        """sets the image agent.
        """
        self._im_agent = pygame.image.load(im_agent_file).convert_alpha()

    def set_im_goal(self, im_goal_file):
        """sets the goal image.
        """
        self._im_goal = pygame.image.load(im_goal_file).convert_alpha()

    @property
    def version(self):
        return self._VER

    @property
    def goal(self):
        return self._goal

    @property
    def is_alive(self):
        return self._alive

    @property
    def rewd_hole(self):
        return self._rewd_hole

    @rewd_hole.setter
    def rewd_hole(self, r):
        self._rewd_hole = r

    @property
    def rewd_wall(self):
        return self._rewd_wall

    @rewd_wall.setter
    def rewd_wall(self, r):
        self._rewd_wall = r

    @property
    def back_color(self):
        return self._back_col

    @back_color.setter
    def back_color(self, color):
        self._back_col = color

    @property
    def grid_color(self):
        return self._grid_col

    @grid_color.setter
    def grid_color(self, color):
        self._grid_col = color

    @property
    def hole_color(self):
        return self._hole_col

    @hole_color.setter
    def hole_color(self, color):
        self._hole_col = color

    @property
    def home_color(self):
        return self._home_col

    @home_color.setter
    def home_color(self, color):
        self._home_col = color

    @property
    def goal_color(self):
        return self._goal_col

    @goal_color.setter
    def goal_color(self, color):
        self._goal_col = color

    def draw_agent(self):
        """draws the agent's image in the current cell.
        """
        x_pos = (self._state % self.ncol) * self._CELL_W
        y_pos = (self._state // self.ncol) * self._CELL_H
        self._scr.blit(self._im_agent, (x_pos, y_pos))

    def draw_grid(self):
        """draws the board grid.
        """
        line_width = 3

        # borde
        pygame.draw.lines(
            self._scr,
            self._grid_col,
            True,
            [(0, 0), (0, self._h), (self._w, self._h), (self._h, 0)],
            line_width)

        # columnas
        for col in range(1, self._env.ncol):
            pygame.draw.line(
                self._scr,
                self._grid_col,
                (col * self._CELL_W, 0), (col * self._CELL_W, self._h),
                line_width)

        # filas
        for row in range(1, self._env.nrow):
            pygame.draw.line(
                self._scr,
                self._grid_col,
                (0, row * self._CELL_H), (self._w, row * self._CELL_H),
                line_width)

    def draw_special_cells(self):
        """draws start, hole and goal cells
        """
        for r in range(self._env.desc.shape[0]):
            for c, val in enumerate(self._env.desc[r]):
                if val == b'S' or val == b'H' or val == b'G':
                    x_pos = c * self._CELL_W
                    y_pos = r * self._CELL_H

                    color = self._home_col if val == b'S' else self._hole_col if val == b'H' else self._goal_col
                    pygame.draw.rect(
                        self._scr,
                        color,
                        pygame.Rect(x_pos, y_pos, self._CELL_W, self._CELL_H)
                    )

                    if self._im_goal and val == b'G':
                        self._scr.blit(self._im_goal, (x_pos, y_pos))

    def init_sound(self):
        # Load sounds
        pygame.mixer.music.load(self._SND_MUSIC)
        self._snd_hit = pygame.mixer.Sound(self._SND_HIT)
        self._snd_fall = pygame.mixer.Sound(self._SND_FALL)
        self._snd_goal = pygame.mixer.Sound(self._SND_GOAL)

        # Start music play
        pygame.mixer.music.play(-1)

    def close(self):
        """close window and exit."""
        pygame.display.quit()
        pygame.quit()
        self._alive = False
        self._done = True
        # exit()

    def run_once(self):
        """executes one iteration of the main display.
        """
        if self._alive:
            self._scr.fill(self._back_col)
            self.draw_special_cells()
            self.draw_grid()
            self.draw_agent()

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
