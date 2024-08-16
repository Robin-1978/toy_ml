models = [
    {
        "name": "lstm_onehot_blue",
        "module" : "DataModel",
        "data" : "load_ssq_blue",
        "model": {
            "module": "model",
            "name": "LSTMOneHotBallModel",
            "param": {
                "input_size": 1,
                "num_classes": 16,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.0,
            },
        },
    },
    {
        "name": "lstm_onehot_red",
        "module" : "DataModel",
        "data" : "load_ssq_red",
        "model": {
            "module": "model",
            "name": "LSTMOneHotBallModel",
            "param": {
                "input_size": 6,
                "num_classes": 33,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.0,
            },
        },
    },
    {
        "name": "lstm_embed_blue",
        "module" : "DataModel",
        "data" : "load_ssq_blue",
        "model": {
            "module": "model",
            "name": "LSTMEmbedBallModel",
            "param": {
                "input_size": 1,
                "num_classes": 16,
                "embedding_size": 16,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.0,
            },
        },
    },
    {
        "name": "lstm_embed_red",
        "module" : "DataModel",
        "data" : "load_ssq_red",
        "model": {
            "module": "model",
            "name": "LSTMEmbedBallModel",
            "param": {
                "input_size": 6,
                "num_classes": 33,
                "embedding_size": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.0,
            },
        },
    },

]
