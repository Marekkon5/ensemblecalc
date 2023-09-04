# ensemblecalc

"Optimizer" calculator for ensembles of AI split vocals.

## Compiling

Install Rust and run: `cargo build --release` (**NOTE:** The `--release` is important otherwise it will be very slow).

Alternativelly you can get the build from Actions tab.


## Usage

### Folder structure:

You should have folders & tracks in the following structure:

```
parent:
  - track_1
    - ref.flac
    - est
      - model_1.flac
      - model_2.flac
      - ...
  
  - track_2
    - ref.flac
    - est
      - model_1.flac
      - model_2.flac
      - ...
```

### Optimizer

The simplest way to run optimizer is:

```
ensemblecalc optimize --path parent --output results.json
```

`output.json` Will then contain the weights in format: `{"filename": weight}` and can be used with ensemble later.

However there is a lot of options to see them run:

```
ensemblecalc optimize --help


Run optimizer on folder of references + estimates

Usage: ensemblecalc optimize [OPTIONS] --path <PATH> --output <OUTPUT>

Options:
  -p, --path <PATH>                  Path to folder of folders
      --ref-name <REF_NAME>          Reference file name [default: ref.flac]
      --est-name <EST_NAME>          Estimate folder name [default: est]
      --sample-rate <SAMPLE_RATE>    [default: 44100]
      --threads <THREADS>            Data loader threads [default: 4]
      --filter <FILTER>              Filenames of estimates to include in calc
      --tol <TOL>                    Solver sd tolerance [default: 0.0005]
      --max-iters <MAX_ITERS>        Max iteration count [default: 500]
      --rng-min <RNG_MIN>            Initial solver rng min value [default: 0]
      --rng-max <RNG_MAX>            Initial solver rng max value [default: 1]
      --extra-params <EXTRA_PARAMS>  Extra layers/params to solver, idk honestly, keep at 1 [default: 1]
      --output <OUTPUT>              Output json path
  -h, --help                         Print help
```


### Ensemble

You can ensemble the tracks using the weights from optimizer. Example:

```
ensemblecalc ensemble --path parent/track_1/est --weights output.json --output output.wav
```

Note that path is to one folder with estimates.

For more options run with `--help`:

```
Ensemble track from weights

Usage: ensemblecalc ensemble [OPTIONS] --path <PATH> --weights <WEIGHTS> --output <OUTPUT>

Options:
  -p, --path <PATH>                Path to estimates
  -w, --weights <WEIGHTS>          Path to weights.json file with format {"name": weight}
  -o, --output <OUTPUT>            Path to .wav file
      --channels <CHANNELS>        [default: 2]
      --sample-rate <SAMPLE_RATE>  [default: 44100]
  -h, --help                       Print help
```