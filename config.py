models = [
    {
        "name": "lstm_onehot_blue",
        "module" : "DataModel",
        "data" : "load_ssq_blue",
        "model": {
            "module": "model",
            "name": "LSTMBallModel",
            "param": {
                "input_size": 19,
                "num_classes": 16,
                "output_size" : 1,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
            },
        },
    },
    # {
    #     "name": "lstm_onehot_red",
    #     "module" : "DataModel",
    #     "data" : "load_ssq_red",
    #     "model": {
    #         "module": "model",
    #         "name": "LSTMBallModel",
    #         "param": {
    #             "input_size": 6,
    #             "num_classes": 33,
    #             "hidden_size": 128,
    #             "num_layers": 2,
    #             "dropout": 0.0,
    #         },
    #     },
    # },

]
