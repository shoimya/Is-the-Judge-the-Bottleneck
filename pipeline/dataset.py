"""HotpotQA / MuSiQue -> Pilot set and Test set question files. Built in T02 (HotpotQA) and T14 (MuSiQue)."""
from pipeline import load_config
from datasets import load_dataset
import os
import json
from pipeline import REPO_ROOT



def main()-> None:
    config = load_config()
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split= "validation")

    pilot_size = 100
    random_seed = config['seeds'][0]
    shuffle_ds = ds.shuffle(seed=random_seed)
    pilot_ds = shuffle_ds.select(range(pilot_size))

    test_size = 1000
    test_ds = shuffle_ds.select(range(pilot_size, pilot_size+test_size))

    out_dir = REPO_ROOT / config["paths"]["data_dir"] / "hotpotqa"
    ids_dir = REPO_ROOT / config["paths"]["results_dir"] / "question_ids"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)



    write_jsonl(pilot_ds, "pilot", out_dir, ids_dir)
    write_jsonl(test_ds, "test", out_dir, ids_dir)






def write_jsonl(ds, qtype, out_file, id_file):

    with open(out_file / f"{qtype}.jsonl", "w", encoding="utf-8") as f:
        for item in ds:
            record = {"id": item["id"], "question": item["question"], "answer": item["answer"], "type": item["type"],
                     "level": item["level"], "gold_titles": get_gold_titles(item)}
            f.write(json.dumps(record) + "\n")
            (id_file / f"hotpotqa_{qtype}.txt").write_text("".join(i + "\n" for i in ds["id"]))



def get_gold_titles(entry):
    gold_titles = list(dict.fromkeys(entry["supporting_facts"]["title"]))
    return gold_titles



if __name__ == "__main__":
    main()