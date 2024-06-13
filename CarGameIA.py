import pygame
from pygame.locals import *
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import imageio

# Initialize the game
pygame.init()

# Set up the screen
width = 500
height = 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Car Game by Audric")

# Set up the colors
gray = (100, 100, 100)
green = (76, 208, 56)
red = (200, 0, 0)
white = (255, 255, 255)
yellow = (255, 232, 0)

# Road and marker sizes
road_width = 300
marker_width = 10
marker_height = 50

# Lane coordinates
left_lane = 150
center_lane = 250
right_lane = 350
lanes = [left_lane, center_lane, right_lane]

# Road and edge markers
road = (100, 0, road_width, height)
left_edge_marker = (95, 0, marker_width, height)
right_edge_marker = (395, 0, marker_width, height)

# For animating movement of the lane markers
lane_marker_move_y = 0

# Player's starting coordinates
player_x = 250
player_y = 400

# Frame settings
clock = pygame.time.Clock()
fps = 60

# Game settings
gameover = False
speed = 5  # Initial speed
score = 0

def scale_image(image, width):
    scale = width / image.get_rect().width
    new_width = int(image.get_rect().width * scale)
    new_height = int(image.get_rect().height * scale)
    return pygame.transform.scale(image, (new_width, new_height))

class Voiture(pygame.sprite.Sprite):
    def __init__(self, image, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = scale_image(image, 45)
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

class PlayerVoiture(Voiture):
    def __init__(self, x, y):
        image = pygame.image.load('photos/car.png')
        super().__init__(image, x, y)

# Sprite groups
player_group = pygame.sprite.Group()
voiture_group = pygame.sprite.Group()

# Create the player's car
player = PlayerVoiture(player_x, player_y)
player_group.add(player)

# Load the Voiture images
image_filenames = ['pickup_truck.png', 'semi_trailer.png', 'taxi.png', 'van.png']
voiture_images = []
for image_filename in image_filenames:
    image = pygame.image.load('photos/' + image_filename)
    voiture_images.append(scale_image(image, 45))

# Load the crash image
crash = pygame.image.load('photos/crash.png')
crash_rect = crash.get_rect()

# Define the DQN agent
class DQNAgent:
    def __init__(self):
        self.state_size = (4,)  # Simplified state: [player_position, car1_position, car2_position, speed]
        self.action_size = 3  # left, right, stay
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the environment
class CarGameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        global player_group, voiture_group, player, speed, score, gameover
        player_group.empty()
        voiture_group.empty()
        player = PlayerVoiture(player_x, player_y)
        player_group.add(player)
        speed = 5  # Reset speed
        score = 0
        gameover = False
        return self.get_state()

    def get_state(self):
        player_position = player.rect.center[0]
        closest_cars = sorted([car.rect.y for car in voiture_group])[:2]
        while len(closest_cars) < 2:
            closest_cars.append(height)
        state = np.array([player_position, closest_cars[0], closest_cars[1], speed])
        return state.reshape(1, -1)

    def step(self, action):
        global player_group, voiture_group, player, speed, score, gameover, lane_marker_move_y

        if action == 0 and player.rect.center[0] > left_lane:  # Move left
            player.rect.x -= 100
        elif action == 1 and player.rect.center[0] < right_lane:  # Move right
            player.rect.x += 100

        lane_marker_move_y += speed * 2
        if lane_marker_move_y >= marker_height * 2:
            lane_marker_move_y = 0

        for voiture in voiture_group:
            voiture.rect.y += speed
            if voiture.rect.top >= height:
                voiture.kill()
                score += 1
                if score % 5 == 0:  # Increase speed every 5 points
                    speed += 1

        if len(voiture_group) < 2:
            add_voiture = True
            for voiture in voiture_group:
                if voiture.rect.top < voiture.rect.height * 1.5:
                    add_voiture = False

            if add_voiture:
                lane = random.choice(lanes)
                image = random.choice(voiture_images)
                new_voiture = Voiture(image, lane, height / -2)
                voiture_group.add(new_voiture)

        reward = 1
        done = False
        if pygame.sprite.spritecollide(player, voiture_group, True):
            gameover = True
            done = True
            reward = -100

        state = self.get_state()
        return state, reward, done

    def render(self):
        screen.fill(green)
        pygame.draw.rect(screen, gray, road)
        pygame.draw.rect(screen, yellow, left_edge_marker)
        pygame.draw.rect(screen, yellow, right_edge_marker)
        for y in range(marker_height * -2, height, marker_height * 2):
            pygame.draw.rect(screen, white, (left_lane + 45, y + lane_marker_move_y, marker_width, marker_height))
            pygame.draw.rect(screen, white, (center_lane + 45, y + lane_marker_move_y, marker_width, marker_height))
        player_group.draw(screen)
        voiture_group.draw(screen)
        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        text = font.render('Score: ' + str(score), True, white)
        text_rect = text.get_rect()
        text_rect.center = (50, 400)
        screen.blit(text, text_rect)
        if gameover:
            screen.blit(crash, crash_rect)
            pygame.draw.rect(screen, red, (0, 50, width, 100))
            text = font.render('Game over. Play again? (Enter Y or N)', True, white)
            text_rect = text.get_rect()
            text_rect.center = (width / 2, 100)
            screen.blit(text, text_rect)
        pygame.display.update()

# Main training loop
env = CarGameEnv()
agent = DQNAgent()
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    time = 0
    frames = []
    while True:  # Continue the episode until the player crashes
        if time % 10 == 0:  # Reduced render frequency for GIF capture
            env.render()
            # Capture frame for GIF
            frame = pygame.surfarray.array3d(screen)
            frames.append(frame)
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1
        if done:
            print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

      # Save the model and GIF at the end of each episode
    agent.model.save(f"dqn_model_{e}.h5")
    # Save GIF
    gif_filename = f'training_episode_{e}.gif'
    imageio.mimsave(gif_filename, frames, fps=15)
    print(f"Saved {gif_filename}")

pygame.quit()
