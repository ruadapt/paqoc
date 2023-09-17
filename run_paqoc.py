#!/usr/bin/env python3
import argparse
import warnings

warnings.filterwarnings("ignore")


def cmdline_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument(
        "-C",
        "--circ",
        type=str,
        required=True,
        help="File name of the optimized circuit (in qasm format).",
    )
    p.add_argument(
        "-D",
        "--pulseDB",
        type=str,
        default="PulseDB/pulse_db.csv",
        help="The local pusle database for storing optimized pulses.",
    )
    p.add_argument(
        "-O",
        "--output",
        type=str,
        default="test.out",
        help="File name of the output result.",
    )
    p.add_argument(
        "-F",
        "--minF",
        type=int,
        default=8,
        help="The minimum appearance of a pattern in the grami graph.",
    )
    p.add_argument(
        "-K",
        "--topK",
        type=int,
        default=5,
        help="The number of selected candidates at each iteration.",
    )
    p.add_argument(
        "-N",
        "--maxN",
        type=int,
        default=3,
        help="The maximum number of qubits allowed in the customized gates.",
    )
    p.add_argument(
        "-G",
        "--maxG",
        type=int,
        default=20,
        help="The maximum number of qubits allowed in the customized gates.",
    )
    p.add_argument(
        "-P",
        "--prune",
        type=str,
        default="critical",
        choices=["critical", "random", "extended"],
        help="The method to generate the candidate customized gates.",
    )
    p.add_argument(
        "-M",
        "--merge",
        type=str,
        default="approx",
        choices=["approx", "random"],
        help="The method to ranking different candidate customized gates.",
    )
    p.add_argument(
        "--nogrami",
        action="store_false",
        help="Disable grami to find frequent subcircuits.",
    )
    p.add_argument(
        "--nopamerge",
        action="store_false",
        help="Disable pamerge to further merge gates.",
    )
    p.add_argument(
        "--protect",
        action="store_true",
        help="Pattern/freuqent subcircuits will not be merged.",
    )
    p.add_argument(
        "--no1qgate",
        action="store_false",
        help="Do not merge continuous single-qubit gates.",
    )
    p.add_argument(
        "--no2qgate",
        action="store_false",
        help="Do not merge continuous two-qubit gates.",
    )
    p.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Print detailed information about this run.",
    )
    return p


def run_paqoc(circ_fname, pulseDB_fname, result_fname, config, details=False):
    from qiskit import IBMQ
    from qiskit.circuit import QuantumCircuit
    from qiskit.compiler import transpile

    import sys
    from os import path
    import pickle
    import pprint

    from helper import generate_2d_grid_coupling
    from paqoc import PAQOC

    fname = path.splitext(path.basename(circ_fname))[0]

    IBMQ.save_account(
        # "PUT YOUR IBMQ ACCOUNT HERE"
        "de38f75d7563f4a91043e92167fd77f665d00f64995c814a029cb1960bc83a93f9d4f59d5ce9581749f5147e8b0ccdc00365263b1462ac73dfc851161ec97a86"
    )
    provider = IBMQ.load_account()
    if sys.version_info.major < 3:
        raise Exception(
            f"Current do not support python version {sys.version_info.major}"
        )
    if sys.version_info.minor > 7:
        simulator_backend = provider.get_backend("ibmq_qasm_simulator")
    else:
        simulator_backend = provider.backend.ibmq_qasm_simulator
    # if 'swap' in simulator_backend._configuration.__dict__['basis_gates']:
    #     simulator_backend._configuration.__dict__['basis_gates'].remove('swap')
    coupling_map = generate_2d_grid_coupling(5)

    circ = QuantumCircuit.from_qasm_file(circ_fname)
    circ.remove_final_measurements()
    if "transpiled" not in circ_fname:
        circ = transpile(
            circ,
            coupling_map=coupling_map,
            backend=simulator_backend,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=2021,
            optimization_level=0,
        )
    if details:
        print(f"file name: {fname}; nqubits: {circ.num_qubits}; ngates: {circ.size()}")
        for name, val in config.items():
            print(f"{name}: {val}; ", end="")
        print()

    paqoc = PAQOC(
        fname,
        circ,
        pulse_db_filename=pulseDB_fname,
        min_freq=config["min_freq"],
        prune_method=config["prune_method"],
        enable_pamerge=config["enable_pamerge"],
        enable_grami=config["enable_grami"],
        proctect_pattern=config["proctect_pattern"],
        merge_cont1q=config["merge_cont1q"],
        merge_cont2q=config["merge_cont2q"],
        merge_heuristic=config["merge_heuristic"],
        topk=config["topk"],
        maxN=config["maxN"],
        maxG=config["maxG"],
    )

    paqoc.run()
    paqoc.save_to_pickle(result_fname)

    with open(result_fname, "rb") as fp:
        results = pickle.load(fp)
    if details:
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        pp.pprint(results)

    return results


if __name__ == "__main__":
    parser = cmdline_args()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit()

    config = {
        "min_freq": args.minF,
        "prune_method": args.prune,
        "topk": args.topK,
        "maxN": args.maxN,
        "maxG": args.maxG,
        "merge_heuristic": args.merge,
        "proctect_pattern": args.protect,
        "merge_cont1q": args.no1qgate,
        "merge_cont2q": args.no2qgate,
        "enable_grami": args.nogrami,
        "enable_pamerge": args.nopamerge,
    }

    circ_fname = args.circ
    pulseDB_fname = args.pulseDB
    result_fname = args.output

    run_paqoc(circ_fname, pulseDB_fname, result_fname, config, args.verbose)
