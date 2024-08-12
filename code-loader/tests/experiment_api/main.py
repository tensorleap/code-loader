from os import path
from code_loader.experiment_api.experiment import init_experiment

# Integration test - make sure you have running tensorleap server

working_dir = path.dirname(path.abspath(__file__))

exp = init_experiment("Exp1", "description", working_dir=working_dir)

h5_file_path = path.join(working_dir, "mnist.h5")

exp.log_epoch(epoch=1, metrics={
    "loss": 0.1,
    "accuracy": 0.9,
    "val_loss": 0.2,
    "val_accuracy": 0.8,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789,
}, model_path=h5_file_path, tags=["best"])
exp.log_epoch(epoch=2, metrics={
    "accuracy": 0.95,
    "val_accuracy": 0.85,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789
}, model_path=h5_file_path)
exp.log_epoch(epoch=3, metrics={
    "accuracy": 0.98,
    "val_accuracy": 0.88,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789
}, model_path=h5_file_path, tags=["latest", "best"])
exp.log_epoch(epoch=4, metrics={
    "accuracy": 0.99,
    "val_accuracy": 0.89,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789
}, model_path=h5_file_path, tags=["latest"])
exp.log_epoch(epoch=5, metrics={
    "accuracy": 0.99,
    "val_accuracy": 0.89,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789
}, model_path=h5_file_path, tags=["latest"])
exp.log_epoch(epoch=6, metrics={
    "accuracy": 0.99,
    "val_accuracy": 0.89,
    "custom_metric": "custom_value",
    "custom_metric2": 123,
    "custom_metric3": 123.456,
    "custom_metric4": 123.456789
}, model_path=h5_file_path, tags=["latest"])


exp.set_notes({ 
    'description': "This is a note", 
    "tags": ["note", "important"],
    "custom_note": "custom_note",
    "nested": {
        "key1": "value1",
        "key2": "value2"
    },
    "nested2": {
        "key1": ["value1", "value2"],
        "key2": "value2"
    },
    "nestedArray": [
        {
            "key1": "value1",
            "key2": "value2"
        },
        {
            "key1": {
                "key1": "value1",
                "key2": [
                    "value1",
                    "value2"
                ]
            },
            "key2": "value2"
        }
    ],
    "arrayInArray": [
        [
            "value1",
            "value2"
        ],
        [
            "value1",
            "value2"
        ],
    ]
})
