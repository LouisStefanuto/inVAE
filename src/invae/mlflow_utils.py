import mlflow


def get_next_run_name(experiment_name: str) -> str:
    """Compute the next run name as run_1, run_2, ..."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Experiment doesn’t exist yet
        return "run_1"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        filter_string="",
    )

    if runs.empty:
        return "run_1"

    # Extract existing run numbers
    existing_numbers = []
    for name in runs["tags.mlflow.runName"]:
        if name.startswith("run_"):
            try:
                n = int(name.split("_")[1])
                existing_numbers.append(n)
            except ValueError:
                continue

    next_number = max(existing_numbers, default=0) + 1
    return f"run_{next_number}"
