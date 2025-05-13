import numpy as np


class MpcControl:

    def __init__(self):
        self._car_image_position = np.array([48, 64])

        # parameter for stanley controller
        # gain for correcting crosstrack error
        self.k_e = 3
        # damping for high speed
        self.k_s = 0.1


    def control(self, trajectory):
        pass

