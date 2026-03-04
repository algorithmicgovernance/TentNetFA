import h5py
import numpy as np
import click
import yaml
import sys


def resample_dataset(inp, out, rng_seed):
    rng = np.random.default_rng(rng_seed)

    with h5py.File(inp, "r") as f:
        labels = f["label"]
        n = labels.shape[0]

        # A tile is "null" if all label pixels are zero
        is_null = np.all(labels[:] == 0, axis=(1, 2))
        is_non_null = ~is_null

        null_idx = np.where(is_null)[0]
        non_null_idx = np.where(is_non_null)[0]

        # keep 25% of null tiles, randomly
        keep_null_n = int(0.25 * len(null_idx))
        keep_null_idx = rng.choice(null_idx, size=keep_null_n, replace=False)

        # keep all non-null tiles
        keep_idx = np.sort(np.concatenate([non_null_idx, keep_null_idx]))

        print(f"Original tiles: {n}")
        print(f"Non-null kept : {len(non_null_idx)}")
        print(f"Null kept     : {len(keep_null_idx)}")
        print(f"Final tiles   : {len(keep_idx)}")

        with h5py.File(out, "w") as g:
            for name, d in f.items():
                g.create_dataset(
                    name,
                    shape=(len(keep_idx), *d.shape[1:]),
                    maxshape=(None, *d.shape[1:]),
                    chunks=d.chunks,
                    dtype=d.dtype,
                    compression=d.compression,
                )
                g[name][:] = d[keep_idx]

    print(f"Saved {out}")


@click.command()
@click.argument("config_path", type=click.Path(exists=True), default="config.yaml")
@click.option("--inp", "cli_inp", help="Input H5 file to reading from.", default=None)
@click.option("--out", "cli_out", help="Output H5 file to write to.", default=None)
@click.option(
    "--seed", "cli_seed", type=int, help="Random seed for sampling.", default=None
)
def cli(config_path, cli_inp, cli_out, cli_seed):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "rebalancing" not in config:
        print(
            f"Error: 'rebalancing' section not found in {config_path}", file=sys.stderr
        )
        sys.exit(1)

    rebal_config = config["rebalancing"]

    inp = cli_inp or rebal_config.get("inp")
    out = cli_out or rebal_config.get("out")
    rng_seed = cli_seed or rebal_config.get("rng_seed")

    if not inp:
        print(
            "Error: 'inp' (input file) missing in 'rebalancing' section and not provided as arg",
            file=sys.stderr,
        )
        sys.exit(1)

    if not out:
        print(
            "Error: 'out' (output file) missing in 'rebalancing' section and not provided as arg",
            file=sys.stderr,
        )
        sys.exit(1)

    resample_dataset(inp, out, rng_seed)


if __name__ == "__main__":
    cli()
