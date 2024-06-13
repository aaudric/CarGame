import pygame
from pygame.locals import *
import random

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

# road and marker sizes
road_width = 300
marker_width = 10
marker_height = 50

# lane coordinates
left_lane = 150
center_lane = 250
right_lane = 350
lanes = [left_lane, center_lane, right_lane]

# road and edge markers
road = (100, 0, road_width, height)
left_edge_marker = (95, 0, marker_width, height)
right_edge_marker = (395, 0, marker_width, height)

# for animating movement of the lane markers
lane_marker_move_y = 0

# player's starting coordinates
player_x = 250
player_y = 400

# frame settings
clock = pygame.time.Clock()
fps = 120

# game settings
gameover = False
speed = 2
score = 0

def scale_image(image, width):
    scale = width / image.get_rect().width
    new_width = int(image.get_rect().width * scale)
    new_height = int(image.get_rect().height * scale)
    return pygame.transform.scale(image, (new_width, new_height))

class Voiture(pygame.sprite.Sprite):
    def __init__(self, image, x, y):
        pygame.sprite.Sprite.__init__(self)

        # scale the image
        pygame.sprite.Sprite.__init__(self)
        self.image = scale_image(image, 45)
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

class PlayerVoiture(Voiture):
    def __init__(self, x, y):
        image = pygame.image.load('CarGame/photos/car.png')
        super().__init__(image, x, y)

# sprite groups
player_group = pygame.sprite.Group()
voiture_group = pygame.sprite.Group()

# create the player's car
player = PlayerVoiture(player_x, player_y)
player_group.add(player)

# load the Voiture images
image_filenames = ['pickup_truck.png', 'semi_trailer.png', 'taxi.png', 'van.png']
voiture_images = []
for image_filename in image_filenames:
    image = pygame.image.load('CarGame/photos/' + image_filename)
    voiture_images.append(scale_image(image, 45))

# load the crash image
crash = pygame.image.load('CarGame/photos/crash.png')
crash_rect = crash.get_rect()

# game loop
running = True
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        # move the player's car using the left/right arrow keys
        if event.type == KEYDOWN:
            if event.key == K_LEFT and player.rect.center[0] > left_lane:
                player.rect.x -= 100
            elif event.key == K_RIGHT and player.rect.center[0] < right_lane:
                player.rect.x += 100

            # check if there's a side swipe collision after changing lanes
            for voiture in voiture_group:
                if pygame.sprite.collide_rect(player, voiture):
                    gameover = True

                    # place the player's car next to other voiture
                    # and determine where to position the crash image
                    if event.key == K_LEFT:
                        player.rect.left = voiture.rect.right
                        crash_rect.center = [player.rect.left, (player.rect.center[1] + voiture.rect.center[1]) / 2]
                    elif event.key == K_RIGHT:
                        player.rect.right = voiture.rect.left
                        crash_rect.center = [player.rect.right, (player.rect.center[1] + voiture.rect.center[1]) / 2]

    # draw the grass
    screen.fill(green)

    # draw the road
    pygame.draw.rect(screen, gray, road)

    # draw the edge markers
    pygame.draw.rect(screen, yellow, left_edge_marker)
    pygame.draw.rect(screen, yellow, right_edge_marker)

    # draw the lane markers
    lane_marker_move_y += speed * 2
    if lane_marker_move_y >= marker_height * 2:
        lane_marker_move_y = 0
    for y in range(marker_height * -2, height, marker_height * 2):
        pygame.draw.rect(screen, white, (left_lane + 45, y + lane_marker_move_y, marker_width, marker_height))
        pygame.draw.rect(screen, white, (center_lane + 45, y + lane_marker_move_y, marker_width, marker_height))

    # draw the player's car
    player_group.draw(screen)

    # add a voiture
    if len(voiture_group) < 2:
        # ensure there's enough gap between voitures
        add_voiture = True
        for voiture in voiture_group:
            if voiture.rect.top < voiture.rect.height * 1.5:
                add_voiture = False

        if add_voiture:
            # select a random lane
            lane = random.choice(lanes)

            # select a random voiture image
            image = random.choice(voiture_images)
            new_voiture = Voiture(image, lane, height / -2)
            voiture_group.add(new_voiture)

    # make the voitures move
    for voiture in voiture_group:
        voiture.rect.y += speed

        # remove voiture once it goes off screen
        if voiture.rect.top >= height:
            voiture.kill()

            # add to score
            score += 1

            # speed up the game after passing 5 voitures
            if score > 0 and score % 5 == 0:
                speed += 1

    # draw the voitures
    voiture_group.draw(screen)

    # display the score
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    text = font.render('Score: ' + str(score), True, white)
    text_rect = text.get_rect()
    text_rect.center = (50, 400)
    screen.blit(text, text_rect)

    # check if there's a head on collision
    if pygame.sprite.spritecollide(player, voiture_group, True):
        gameover = True
        crash_rect.center = [player.rect.center[0], player.rect.top]

    # display game over
    if gameover:
        screen.blit(crash, crash_rect)

        pygame.draw.rect(screen, red, (0, 50, width, 100))

        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        text = font.render('Game over. Play again? (Enter Y or N)', True, white)
        text_rect = text.get_rect()
        text_rect.center = (width / 2, 100)
        screen.blit(text, text_rect)

    pygame.display.update()

    # wait for user's input to play again or exit
    while gameover:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == QUIT:
                gameover = False
                running = False

            # get the user's input (y or n)
            if event.type == KEYDOWN:
                if event.key == K_y:
                    # reset the game
                    gameover = False
                    speed = 2
                    score = 0
                    voiture_group.empty()
                    player.rect.center = [player_x, player_y]
                elif event.key == K_n:
                    # exit the loops
                    gameover = False
                    running = False

pygame.quit()
