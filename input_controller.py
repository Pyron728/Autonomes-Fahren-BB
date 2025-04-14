import pygame


class InputController:
    """
    Input controller for the car. It can be controlled by keyboard or joystick/steering
    wheel.

    The car can be controlled by the following keys:
        - W or UP: Accelerate
        - S or DOWN: Brake
        - A or LEFT: Steer left
        - D or RIGHT: Steer right
        - ESC: Quit
        - SPACE: Skip the current run

    The game can be controlled by the following joystick/steering wheel buttons:
        - Button 6 or 9: Quit
        - Button 7 or 10: Skip the current run
    """

    def __init__(self):
        self.skip = False
        self.quit = False

        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

        # Initialize joysticks if they are available
        pygame.joystick.init()
        self.joystick = [
            pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())
        ]

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

            if event.type == pygame.JOYBUTTONDOWN:
                self.button_press(event)

        if len(self.joystick) > 0:
            self.accelerate = ((self.joystick[0].get_axis(5) + 1) / 2) ** 2
            self.brake = ((self.joystick[0].get_axis(2) + 1) / 2) ** 2 * 0.75
            self.steer = self.joystick[0].get_axis(0) ** 3
        else:
            keys = pygame.key.get_pressed()
            self.accelerate = 0.5 if keys[pygame.K_w] or keys[pygame.K_UP] else 0
            self.brake = 0.8 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0
            self.steer = (
                1
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]
                else (-1 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0)
            )

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:
            self.quit = True
        if event.key == pygame.K_SPACE:
            self.skip = True

    def button_press(self, event):
        if event.button == 6 or event.button == 9:
            self.quit = True
        if event.button == 7 or event.button == 10:
            self.skip = True
