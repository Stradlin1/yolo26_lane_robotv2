from ultralytics import YOLO
from ultralytics.cfg import get_cfg


def main():
    default_yaml = "ultralytics/cfg/default.yaml"

    # 读取 default.yaml
    args = get_cfg(default_yaml)

    # 这些字段应当已经在 default.yaml 里改好
    # task: ufld
    # mode: train
    # model: ultralytics/cfg/models/26/yolo26-ufld.yaml
    # data: ultralytics/cfg/datasets/lane-ufld.yaml
    model = YOLO(args.model, task=args.task)

    # 直接把 default.yaml 里的配置传给 train
    model.train(cfg=default_yaml)


if __name__ == "__main__":
    main()