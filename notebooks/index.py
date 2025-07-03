from infonet import infer


config_path = "../configs/infonet/config.yaml"
model = infer.create_model(
    config=config_path,
)
print(model.device)