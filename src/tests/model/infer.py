import os
import re
from subprocess import PIPE, Popen, TimeoutExpired

def get_query_list(file_path: str) -> list:
    return [f for f in os.listdir(file_path)]

def run_query(query_path: str, model_path: str) -> str:
    try:
        process = Popen([
            "/var/www/html/wayang-assembly/target/wayang-0.7.1/bin/wayang-submit",
            "org.apache.wayang.ml.benchmarks.Inference",
            "java,spark,flink,postgres file:///opt/data/ /var/www/html/data/",
            query_path,
            model_path
        ], stdout=PIPE, stderr=PIPE, start_new_session=True)

        record_plan = False
        plan_str = ""
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if "DONE" in line.strip():
                break

            if record_plan:
                operator = re.sub(r"id=[^,\]]+", "id=0", line.strip())
                plan_str += f"{operator}\n"
            if "== Execution Plan ==" in line.strip():
                record_plan = True

        print(plan_str)
        return plan_str
    except Exception as e:
        print(e)


def main ():
    query_dir: str = "/var/www/html/wayang-plugins/wayang-ml/src/main/resources/benchmarks/job/light"
    queries: list = get_query_list(query_dir)
    for query in queries:
        query_path: str = f"{query_dir}/{query}"
        native_plan: str = run_query(query_path, "")
        model_plan: str = run_query(query_path, "bvae /var/www/html/wayang-plugins/wayang-ml/src/main/python/python-ml/src/Models/imdb/bvae-1.onnx /var/www/html/data/")

        assert native_plan == model_plan


if __name__ == "__main__":
    main()

