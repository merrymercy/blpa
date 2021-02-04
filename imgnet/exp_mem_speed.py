import os
import json
import argparse

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


alg_to_config = {
    "exact": "-qtype 0",
    "quantize": "-qtype 4",
}

network_to_batch_size = {
    "resnet152": [128, 1024],
}

network_to_command = {
    "resnet152":  "python3 train_152.py -save saved_models -bs BS CONFIG -model resnet152",
    "resnet50":  "python3 train_152.py -save saved_models -bs BS CONFIG -model resnet50",
    "wide_resnet101_2":  "python3 train_152.py -save saved_models -bs BS CONFIG -model wide_resnet101_2",
}


def run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=False):
    os.environ['DEBUG_MEM'] = str(debug_mem)
    os.environ['DEBUG_SPEED'] = str(debug_speed)
    batch_size = batch_size - batch_size % 2
    cmd = network_to_command[network]
    cmd = cmd.replace("BS", f"{batch_size}").replace("CONFIG", alg_to_config[alg])
    ret_code = run_cmd(cmd)

    if ret_code != 0:
        out_file = "speed_results.tsv"
        with open(out_file, "a") as fout:
            val_dict = {
                "network": network,
                "algorithm": alg,
                "batch_size": batch_size,
                "ips": -1,
            }
            fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")

    return ret_code


def binary_search_max_batch(network, alg, low, high):
    ret = 0

    while low <= high:
        mid = low + (high - low) // 2
        success = run_benchmark(network, alg, mid, debug_speed=True) == 0
        if success:
            ret = mid
            low = mid + 1
        else:
            high = mid - 1

    return ret


def get_ips(network, alg, batch_size):
    run_benchmark(network, alg, batch_size, debug_speed=True)
    line = list(open("speed_results.tsv").readlines())[-1]
    return json.loads(line)['ips']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
            choices=['linear_scan', 'binary_search'],
            default='linear_scan')
    args = parser.parse_args()

    networks = ['resnet152', 'resnet50', 'wide_resnet101_2']
    algs = ['quantize']
    batch_sizes = list(range(32, 1024, 32))

    if args.mode == 'linear_scan':
        for network in networks: 
            for alg in algs:
                for batch_size in (batch_sizes or network_to_batch_size[network]):
                    if run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False) != 0:
                        break
    elif args.mode == 'binary_search':
        for network in networks:
            for alg in algs:
                low, high = network_to_batch_size[network][0], network_to_batch_size[network][-1]
                max_batch_size = binary_search_max_batch(network, alg, low, high)
                ips = get_ips(network, alg, max_batch_size)

                out_file = "max_batch_results.tsv"
                with open(out_file, "a") as fout:
                    val_dict = {
                        "network": network,
                        "algorithm": alg,
                        "max_batch_size": max_batch_size,
                        "ips": ips,
                    }
                    fout.write(json.dumps(val_dict) + "\n")
                print(f"save results to {out_file}")

