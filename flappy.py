import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
from collections import deque
import cv2


class Flappy(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, width=288, height=512, render_mode='rgb_array'):
        super().__init__()
        pygame.init()

        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.screen = pygame.Surface((self.width, self.height))
        self.window = None
        self.action_space = spaces.Discrete(2)

        # stacked grayscale frames (C=4, H=84, W=84)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

        # bird
        self.bird_x = int(self.width * 0.2)
        self.bird_y = int(self.height / 2)
        self.bird_radius = 12
        self.gravity = 0.5
        self.flap_velocity = -8
        self.bird_velocity = 0

        # pipes
        self.pipe_width = 52
        self.pipe_gap = 100
        self.pipe_speed = 4
        self.pipes = []
        self.passed_pipes = set()  # Track which pipes have been passed

        # state / rendering
        self.score = 0
        self.done = False
        self.clock = pygame.time.Clock()

        # frame stack
        self.frames = deque(maxlen=4)

        # initial pipe(s)
        self.spawn_pipe()

    def spawn_pipe(self):
        gap_y = np.random.randint(int(self.pipe_gap), self.height - int(self.pipe_gap))
        pipe_top = pygame.Rect(self.width, 0, self.pipe_width, gap_y - self.pipe_gap // 2)
        pipe_bottom = pygame.Rect(self.width, gap_y + self.pipe_gap // 2, self.pipe_width,
                                  self.height - (gap_y + self.pipe_gap // 2))
        pipe_id = len(self.pipes)  # Simple ID system
        self.pipes.append((pipe_top, pipe_bottom, pipe_id))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bird_y = int(self.height / 2)
        self.bird_velocity = 0
        self.pipes = []
        self.passed_pipes = set()
        self.spawn_pipe()
        self.score = 0
        self.done = False
        self.frames.clear()

        obs = self.get_observation(init_stack=True)
        return obs, {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0.0, True, False, {}

        reward = 0.0

        # action - removed flap penalty
        if action == 1:
            self.bird_velocity = self.flap_velocity
        
        # physics
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        # move pipes
        new_pipes = []
        for top, bottom, pipe_id in self.pipes:
            top.x -= self.pipe_speed
            bottom.x -= self.pipe_speed
            # keep only on-screen pipes
            if top.right > 0:
                new_pipes.append((top, bottom, pipe_id))
            else:
                # Clean up passed pipe tracking when pipe goes off screen
                self.passed_pipes.discard(pipe_id)
        self.pipes = new_pipes

        # spawn new pipe if needed
        if len(self.pipes) == 0 or self.pipes[-1][0].x < self.width - 200:
            self.spawn_pipe()

        # collisions
        bird_rect = pygame.Rect(self.bird_x - self.bird_radius, int(self.bird_y) - self.bird_radius,
                                self.bird_radius * 2, self.bird_radius * 2)
        
        # boundary collisions
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.height:
            self.done = True
        
        # pipe collisions
        for top, bottom, pipe_id in self.pipes:
            if bird_rect.colliderect(top) or bird_rect.colliderect(bottom):
                self.done = True

        # reward calculation
        if self.done:
            reward = -10.0
        else:
            # base survival reward
            reward = 0.1
            
            # small forward progress reward
            reward += 0.01
            
            if self.pipes:
                closest_pipe = self.pipes[0]
                pipe_center_x = closest_pipe[0].centerx
                pipe_id = closest_pipe[2]
                
                # check if bird just passed pipe (give bonus only once)
                if (self.bird_x > pipe_center_x and 
                    pipe_id not in self.passed_pipes):
                    reward += 10.0
                    self.passed_pipes.add(pipe_id)
                    self.score += 1
                
                # positioning bonus - reward for staying near pipe center
                pipe_center_y = (closest_pipe[0].bottom + closest_pipe[1].top) / 2
                y_distance = abs(self.bird_y - pipe_center_y)
                positioning_bonus = max(0, (self.pipe_gap/2 - y_distance) / (self.pipe_gap/2))
                reward += positioning_bonus * 0.5  # Scale down positioning bonus
        
        obs = self.get_observation()
        terminated = self.done
        truncated = False
        info = {"score": self.score}
        return obs, reward, terminated, truncated, info

    def draw_scene(self):
        """Separate method to draw the scene"""
        self.screen.fill((135, 206, 235))  # sky blue
        
        # draw pipes
        for top, bottom, _ in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), top)
            pygame.draw.rect(self.screen, (0, 255, 0), bottom)
        
        # draw bird
        pygame.draw.circle(self.screen, (255, 255, 0), 
                          (int(self.bird_x), int(self.bird_y)), self.bird_radius)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def get_observation(self, init_stack=False):
        # draw scene to get current frame
        self.draw_scene()
        
        # convert surface to RGB array
        rgb = pygame.surfarray.array3d(self.screen)  # (W, H, C)
        rgb = np.transpose(rgb, (1, 0, 2))  # (H, W, C)

        processed = self.preprocess_frame(rgb)  # (84,84), uint8

        # initialize or push to deque
        if init_stack or len(self.frames) == 0:
            self.frames.clear()
            for _ in range(4):
                self.frames.append(processed)
        else:
            self.frames.append(processed)

        # return (C, H, W)
        stacked = np.stack(self.frames, axis=0).astype(np.uint8)
        return stacked

    def render(self):
        if self.render_mode == 'human':
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Flappy Bird")
            
            # draw scene if not already drawn
            self.draw_scene()
            
            # blit to window and display
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
        elif self.render_mode == 'rgb_array':
            # draw scene and return RGB array
            self.draw_scene()
            rgb = pygame.surfarray.array3d(self.screen)
            return np.transpose(rgb, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()