import argparse


def get_parser() -> argparse.ArgumentParser:
    # Set Argument Parser
    parser = argparse.ArgumentParser(
        prog="Prima Power Project",
        description="Program to train and evaluate Models",
        epilog="Thanks for running me :-)",
    )

    parser.add_argument(
        "--hugging_face_token",
        type=str,
        required=True,
        default="hugging_face_token",
        help="Token to download hugging face models",
    )

    parser.add_argument(
        "--groq_api_key",
        type=str,
        help="Token to use Groq Models",
    )

    parser.add_argument(
        "--framework",
        type=str,
        help="Type of framework used for experiment (e.g. sick, sick++, few_shot, etc.)",
    )

    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--test_dataset", type=str, help="Test Dataset file path")
    parser.add_argument("--train_dataset", type=str, help="Train Dataset file path")

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="name of the experiment. It will be used to create the folder containing all the run results and model weights",
    )

    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        help="Phase of the experiment",
    )

    parser.add_argument(
        "--not_use_local_logging",
        action="store_true",
        help="disable experiment track with build-in serializer",
    )

    parser.add_argument(
        "--not_use_wandb",
        action="store_true",
        help="disable experiment track with wandb logging",
    )

    parser.add_argument("--seed", type=int, default=516)

    # Training hyperparameters
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=20)
    # parser.add_argument('--display_step',type=int, default=2000)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    # Model hyperparameters
    # parser.add_argument('--model_name',type=ModelCheckpointOptions, choices=list(ModelCheckpointOptions), default='facebook/bart-large-xsum')
    parser.add_argument(
        "--model_name",
        type=str,
    )

    # Few-shot params
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--is_llm", type=bool, default=False)
    parser.add_argument("--answers_folder", type=str)

    # Inference params
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="./new_weights_comet/final_Trial1_context_comet",
    )
    parser.add_argument("--test_output_file_name", type=str, default="results.json")

    return parser
