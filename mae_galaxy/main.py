import argparse

def main():
    parser = argparse.ArgumentParser(description="MAE Galaxy Project Entry Point")
    parser.add_argument("command", choices=["help"], nargs="?", default="help")
    args = parser.parse_args()

    if args.command == "help":
        print("Use module entry points:")
        print("  python -m mae_galaxy.training.train_mae --help")
        print("  python -m mae_galaxy.training.linear_probe --help")
        print("  python -m mae_galaxy.experiments.run_experiments --help")


if __name__ == "__main__":
    main()









