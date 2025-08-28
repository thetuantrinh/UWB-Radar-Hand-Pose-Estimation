import json
import io
import inspect
import argparse
import time
import numpy as np

text_colors = {
    "red": "\033[31m",
    "bold": "\033[1m",
    "end_color": "\033[0m",
    "light_red": "\033[36m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
}


def export_training_history(
    model="",
    scheduler="",
    optimizer="",
    criterion="",
    epochs="",
    learning_rate="",
    batch_size="",
    train_ratio="",
    history="",
    dataset="",
    model_saved_file="",
    saved_file="history/training_history.json",
    train_img_files="",
    val_img_files="",
):
    def object2str(object_var):
        output = io.StringIO()
        print(object_var, file=output)
        return str(output.getvalue())

    model = model.to("cpu")
    total_params = sum(p.numel() for p in model.parameters())

    saved_dic = {
        "model": inspect.getsource(type(model)),
        "model_arch": object2str(model),
        "total_params": total_params,
        "scheduler": {
            "optimizer": object2str(scheduler.optimizer),
            "factor": scheduler.factor,
            "patience": scheduler.patience,
            "min_lrs": scheduler.min_lrs,
        },
        "criterion": object2str(criterion),
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "epochs": epochs,
        "init_learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "dataset": dataset,
        "history": history,
        "model_saved_file": model_saved_file,
        "train_img_files": train_img_files,  # sample train batch
        "val_img_files": val_img_files,  # sample val batch
    }

    data = []

    try:
        with open(saved_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass

    data.append(saved_dic)
    time.sleep(np.random.random() * 5)  # delay time for writing

    with open(saved_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully appended to {saved_file}")


def get_trained_model(
    json_file="json_file.json", min_val=False, train_val_diff=True, idx=""
):
    with open(json_file) as f:
        data = json.load(f)

    if min_val:
        min_losses = []
        for exp in data:
            min_losses.append(exp["history"]["min_loss"])
        idx = min_losses.index(min(min_losses))
        print(
            f"min_loss in all run experience is {min(min_losses)} at exp {idx}"
        )
        print("--------------------------------------------")
        print(data[idx]["model"])
        print("--------------------------------------------")
        print(f"batch size {data[idx]['batch_size']}")
        print("--------------------------------------------")
        print(
            f"min_loss {data[idx]['history']['min_loss']} min_train_loss {min(data[idx]['history']['train_loss'])}"
        )
        print("--------------------------------------------")
        print(f"loss function {data[idx]['criterion']}")
    elif train_val_diff:
        min_train_val_diff = []
        min_val_loss = []
        min_train_loss = []
        for idx, exp in enumerate(data):
            train_loss = exp["history"]["train_loss"]
            val_loss = exp["history"]["val_loss"]
            min_train_loss.append(min(train_loss))
            min_val_loss.append(min(val_loss))
            min_train_val_diff.append(min(val_loss) - min(train_loss))
            print(
                f"train val discrepancy = {min_train_val_diff[-1]}, "
                + f" min_train_loss {min_train_loss[-1]}, "
                + f"min_val_loss {min_val_loss[-1]}, "
                + f"batch size {exp['batch_size']} "
                + "from idx {idx}"
            )
        print("-------------------------------------------")

        min_train_loss_dif_idx = min_train_val_diff.index(
            min(min_train_val_diff)
        )
        print(
            f"min train val discrepancy = {highlight(min(min_train_val_diff))}, "
            + f"min_train_loss {min_train_loss[min_train_loss_dif_idx]}, "
            + f"min_val_loss {min_val_loss[min_train_loss_dif_idx]} "
            + f"batch size {exp['batch_size']} "
            + f"from idx {min_train_loss_dif_idx}"
        )

        min_val_loss_idx = min_val_loss.index(min(min_val_loss))

        print(
            f"train val discrepancy = {min_train_val_diff[min_val_loss_idx]}, "
            + f"min_train_loss {min_train_loss[min_val_loss_idx]}, "
            + f"min_val_loss {highlight(min_val_loss[min_val_loss_idx])}, "
            + f"batch size {exp['batch_size']} "
            + f"from idx {min_val_loss_idx}"
        )

    elif idx:
        model_arch = data[idx]["model_arch"]
        model_def = data[idx]["model"]
        batch_size = data[idx]["batch_size"]
        dataset = data[idx]["dataset"]
        loss = data[idx]["criterion"]
        lr = data[idx]["init_learning_rate"]
        print("----------------------------------------------")
        print(model_arch)
        print("----------------------------------------------")
        print(model_def)
        print("----------------------------------------------")
        print(
            f"batch size {batch_size} dataset {dataset} loss function {loss} lr {lr}"
        )


def highlight(text, highlight_color="yellow"):
    colored_text = (
        text_colors[highlight_color]
        + text_colors["bold"]
        + str(text)
        + text_colors["end_color"]
    )
    return colored_text


def parser():
    parser = argparse.ArgumentParser(
        description="Finding smallest trained val_loss"
    )
    parser.add_argument("--json-file", default="path", type=str)
    parser.add_argument(
        "--min-val", action="store_true", help="print min val_loss"
    )
    parser.add_argument(
        "--train-val-diff",
        action="store_true",
        help="print discrepancy between val and train loss",
    )
    parser.add_argument("--export-model_idx", action="store_true")
    parser.add_argument("--idx", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    get_trained_model(
        json_file=args.json_file,
        min_val=args.min_val,
        train_val_diff=args.train_val_diff,
        idx=args.idx,
    )
