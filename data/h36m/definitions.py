TOTAL_SAMPLES = 52411  # 52411 is the total number of samples of 3D joints

JOINT_NUM = 32

VAR_NAMES = ['a', 'b', 'c', 'couple', 'final',
             'legs', 'load', 'lower_arms', 'neck', 'rapid',
             'repeated', 'risk', 'sit', 'static', 'support',
             'trunk', 'unstable', 'upper_arms', 'wrists']

CLASS_NUMS = [12, 12, 12, 4, 15,
              4, 3, 2, 3, 2,
              2, 5, 2, 2, 2,
              5, 2, 6, 3]

CLASS_BIASES = [1, 1, 1, 0, 1,
                1, 0, 1, 1, 0,
                0, 1, 0, 0, -1,
                1, 0, 1, 1]

# the joints used in calculating the REBA risks of specific body parts
JOINTS_OF_PARTS = {5: [1, 2, 3, 6, 7, 8],
                   7: [17, 18, 19, 25, 26, 27],
                   8: [12, 13, 14, 15, 17, 25],
                   15: [0, 1, 6, 13, 17, 25],
                   17: [12, 17, 18, 13, 25, 26],
                   18: [17, 18, 19, 21, 22, 25, 26, 27, 29, 30]}

JOINT_BOX_SIZE = 15

VIEWS = {'front': 1,
         'side': 2,
         'back': 3,
         'all': None}
