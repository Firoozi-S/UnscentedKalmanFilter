import numpy as np

INTEGRATION_STEP = 0.05
K = np.linspace(0, 200, int(200 // INTEGRATION_STEP))

TRUE_INITIAL_CONDITION = [6500.4,
                          349.14,
                          -1.8093,
                          -6.7967,
                          0.6932
                        ]

TRUE_INITIAL_COV = [[10**-6, 0, 0, 0, 0],
                    [0, 10**-6, 0, 0, 0],
                    [0, 0, 10**-6, 0, 0],
                    [0, 0, 0, 10**-6, 0],
                    [0, 0, 0, 0, 0]
                ]


INITIAL_STATE = np.array([[6500.4,
                349.14,
                -1.8093,
                -6.7967,
                0
                ]])

INITIAL_COV = np.array([[10**-6, 0, 0, 0, 0],
                [0, 10**-6, 0, 0, 0],
                [0, 0, 10**-6, 0, 0],
                [0, 0, 0, 10**-6, 0],
                [0, 0, 0, 0, 1]
            ])

PROCESS_NOISE_MEAN = np.array([[0.,
                      0.,
                      0.
                    ]])

PROCESS_NOISE_COV = np.array([[2.4064*(10**-5), 0, 0],
                     [0, 2.4064*(10**-5), 0],
                     [0, 0, 0]
                     ])


