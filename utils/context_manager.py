import json

def store_task_status(task_id, status, path="task_memory.json"):
    try:
        with open(path, "r") as f:
            db = json.load(f)
    except FileNotFoundError:
        db = {}

    db[task_id] = status
    with open(path, "w") as f:
        json.dump(db, f, indent=2)

