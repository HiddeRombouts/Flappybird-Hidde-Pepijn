import pygame
import random
import os
import time
import neat
import pickle
import sys
pygame.font.init()  # init font

# ---------- SPEL / CONSTANTEN ----------
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
FPS = 100

PIPE_GAP = 160
PIPE_VEL = 5
BASE_VEL = 5

STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird (NEAT)")

# images (zorg dat imgs/ aanwezig is)
pipe_img = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(
    pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(),
    (600, 900))
bird_images = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png")))
    for x in range(1, 4)
]
base_img = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

gen = 0
TARGET_SCORE = 1000
WINNER_PICKLE = "winner.pkl"


# ---------- OBJECTEN ----------
class Bird:
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel * (self.tick_count) + 0.5 * (3) * (self.tick_count)**2

        if displacement >= 16:
            displacement = (displacement / abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe():
    GAP = PIPE_GAP
    VEL = PIPE_VEL

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        return b_point or t_point


class Base:
    VEL = BASE_VEL
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2,
                                  bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2,
                                  pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2,
                                  bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2,
                                  pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    score_label = STAT_FONT.render("Gens: " + str(gen - 1), 1, (255, 255, 255))
    win.blit(score_label, (10, 10))
    score_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(score_label, (10, 50))
    pygame.display.update()


# ---------- EVALUATIE / FITNESS ----------
def eval_genomes(genomes, config):
    """
    Run one generation. We use score (number of pipes passed) as the core fitness.
    Stop-and-save when score >= TARGET_SCORE.
    """
    global WIN, gen
    win = WIN
    gen += 1

    nets = []
    birds = []
    ge = []

    # create nets, birds and genome-list
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()
    run = True
    while run and len(birds) > 0:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # determine which pipe to use for inputs
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # loop through birds
        for x, bird in enumerate(birds):
            # small fitness for staying alive each frame
            ge[x].fitness += 0.05
            bird.move()

            # inputs to the NN: normalize values to reasonable scale
            # bird.y (0..WIN_HEIGHT), distance to top gap, distance to bottom gap, bird vertical velocity
            try:
                inputs = (
                    bird.y / WIN_HEIGHT,
                    (pipes[pipe_ind].height - bird.y) / WIN_HEIGHT,
                    (pipes[pipe_ind].bottom - bird.y) / WIN_HEIGHT,
                    (bird.vel + 20) / 40.0  # normalized vel approx
                )
            except Exception:
                inputs = (bird.y / WIN_HEIGHT, 0.5, 0.5, (bird.vel + 20) / 40.0)

            output = nets[x].activate(inputs)
            if output[0] > 0.5:
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for b_i, bird in enumerate(birds):
                if pipe.collide(bird, win):
                    ge[b_i].fitness -= 1
                    # remove dead bird
                    nets.pop(b_i)
                    ge.pop(b_i)
                    birds.pop(b_i)
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            # scoring: when a bird passes a pipe
            # note: we only want to increment score once per pipe
            if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # reward remaining genomes
            for genome in ge:
                genome.fitness += 5

            pipes.append(Pipe(WIN_WIDTH))

            # Check if target achieved
            if score >= TARGET_SCORE:
                # pick the genome with highest fitness among survivors
                if len(ge) > 0:
                    best = max(ge, key=lambda g: g.fitness)
                    with open(WINNER_PICKLE, "wb") as f:
                        pickle.dump(best, f)
                    print(f"TARGET reached! Score {score}. Winner saved to {WINNER_PICKLE}")
                pygame.quit()
                # raise an exception to stop the NEAT run cleanly outside
                raise Exception("TARGET_REACHED")

        for r in rem:
            if r in pipes:
                pipes.remove(r)

        # remove birds that hit floor or go offscreen
        for b_i, bird in enumerate(list(birds)):
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                # remove from lists
                try:
                    nets.pop(b_i)
                    ge.pop(b_i)
                    birds.pop(b_i)
                except:
                    pass

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)


# ---------- RUN NEAT ----------
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Optional: override fitness_threshold from config file
    # so p.run would stop if a genome fitness >= this threshold
    # config.fitness_threshold = TARGET_SCORE

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    try:
        # allow up to 200 generations as a (large) upper limit
        winner = p.run(eval_genomes, 200)
    except Exception as e:
        if str(e) == "TARGET_REACHED":
            print("Training halted: target score reached and winner saved.")
            # load and return the saved winner
            with open(WINNER_PICKLE, "rb") as f:
                winner = pickle.load(f)
            return winner, config
        else:
            raise

    # Normal end (if 200 gens finished)
    print('\nBest genome from run:\n{!s}'.format(winner))
    with open(WINNER_PICKLE, "wb") as f:
        pickle.dump(winner, f)
    print(f"Final winner saved to {WINNER_PICKLE}")
    return winner, config


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    winner, config = run(config_path)
    pygame.quit()
