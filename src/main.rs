#[macro_use] extern crate log;
#[macro_use] extern crate anyhow;

use anyhow::Error;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver};
use kdam::{Column, RichProgress, tqdm, BarExt};
use std::collections::HashMap;
use std::fs::File;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Executor, CostFunction, State};
use argmin::solver::neldermead::NelderMead;
use hound::{WavWriter, WavSpec, SampleFormat};
use rand::Rng;
use rodio::Decoder;
use rodio::source::UniformSourceIterator;
use threadpool::ThreadPool;
use rayon::prelude::*;

mod gui;

fn main() {
    std::env::set_var("RUST_LOG", "ensemblecalc=debug,info");
    pretty_env_logger::init();

    run(Cli::parse()).expect("Failed");
}

#[derive(Parser, Debug)]
pub enum Cli {
    GUI {},

    /// Run optimizer on folder of references + estimates
    Optimize {
        /// Path to folder of folders
        #[arg(long, short)]
        path: PathBuf,

        /// Reference file name
        #[arg(long, default_value_t = String::from("ref.flac"))]
        ref_name: String,

        /// Estimate folder name
        #[arg(long, default_value_t = String::from("est"))]
        est_name: String,

        #[arg(long, default_value_t = 44100)]
        sample_rate: u32,

        /// Data loader threads
        #[arg(long, default_value_t = 4)]
        threads: usize,

        /// Filenames of estimates to include in calc
        #[arg(long)]
        filter: Vec<String>,

        /// Solver sd tolerance
        #[arg(long, default_value_t = 0.0005)]
        tol: f32,

        /// Max iteration count
        #[arg(long, default_value_t = 500)]
        max_iters: u64,

        /// Initial solver rng min value
        #[arg(long, default_value_t = 0.0)]
        rng_min: f32,

        /// Initial solver rng max value
        #[arg(long, default_value_t = 1.0)]
        rng_max: f32,

        /// Extra layers/params to solver, idk honestly, keep at 1
        #[arg(long, default_value_t = 1)]
        extra_params: usize,

        /// Output json path
        #[arg(long)]
        output: PathBuf,
    },

    /// Ensemble track from weights
    Ensemble {
        /// Path to estimates
        #[arg(long, short)]
        path: PathBuf,

        /// Path to weights.json file with format {"name": weight}
        #[arg(long, short)]
        weights: PathBuf,

        /// Path to .wav file
        #[arg(long, short)]
        output: PathBuf,

        #[arg(long, default_value_t = 2)]
        channels: u16,

        #[arg(long, default_value_t = 44100)]
        sample_rate: u32
    }

}

fn run(cli: Cli) -> Result<(), Error> {
    match cli {
        Cli::GUI {} => {
            gui::start_gui();
            return Ok(())
        }
        Cli::Optimize { path, ref_name, est_name, sample_rate, threads, filter, tol, max_iters, output, rng_min, rng_max, extra_params } => {
            let data = load_data(path, filter, threads, &ref_name, &est_name, sample_rate)?;
            match optimize_sets(data, tol, max_iters, extra_params, rng_min..=rng_max) {
                Some(r) => {
                    std::fs::write(output, serde_json::to_string_pretty(&r)?)?;
                },
                None => error!("Optimizer returned no result!"),
            }
        },
        Cli::Ensemble { path, weights, output, channels, sample_rate } => {
            let weights: HashMap<String, f32> = serde_json::from_str(&std::fs::read_to_string(&weights)?)?;
            let ensemble = ensemble_dir(path, weights, sample_rate)?;
            write_wav(output, &ensemble, channels, sample_rate)?;
        },
        
    }

    Ok(())
}

/// Ensemble dir of files
pub fn ensemble_dir(path: impl AsRef<Path>, weights: HashMap<String, f32>, sample_rate: u32) -> Result<Vec<f32>, Error> {
    let mut files = vec![];
    for e in std::fs::read_dir(&path)?.into_iter().filter_map(|e| e.ok()) {
        if let Some(weight) = weights.get(&e.file_name().to_string_lossy().to_string()) {
            let data = decode_file(e.path(), sample_rate)?;
            files.push((data, *weight));
        }
    }
    let inputs = files.iter().map(|(d, w)| (d, *w)).collect::<Vec<_>>();
    Ok(ensemble(&inputs))

}

pub fn load_data_iter(
    path: impl AsRef<Path>, 
    filter: Vec<String>, 
    threads: usize,
    ref_name: &str,
    est_name: &str,
    sample_rate: u32
) -> Result<(usize, Receiver<TrackSet>), Error> {
    // Get list of entries
    let entries = std::fs::read_dir(&path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.path().join(ref_name).exists() && e.path().join(est_name).exists())
        .collect::<Vec<_>>();

    // Create thread pool
    let pool = ThreadPool::new(threads);
    let (tx, rx) = unbounded();
    let count = entries.len();
    let filter = Arc::new(filter);

    // Start tasks
    for entry in entries {
        let tx = tx.clone();
        let path = entry.path();
        let ref_path = path.join(ref_name);
        let est_path = path.join(est_name);
        let filter = filter.clone();
        pool.execute(move || {
            match TrackSet::load(ref_path, est_path, &filter, sample_rate) {
                Ok(t) => tx.send(t).unwrap(),
                Err(e) => error!("Failed loading {path:?}: {e}"),
            }
        });
    }
    Ok((count, rx))
}

/// Load data
pub fn load_data(
    path: impl AsRef<Path>, 
    filter: Vec<String>, 
    threads: usize,
    ref_name: &str,
    est_name: &str,
    sample_rate: u32
) -> Result<TrackSetSet, Error> {
   let (count, rx) = load_data_iter(path, filter, threads, ref_name, est_name, sample_rate)?;

    // Progress bar
    let mut pb = RichProgress::new(
        tqdm!( total = count ),
        vec![
            Column::Animation,
            Column::Percentage(1),
            Column::Text("•".to_owned()),
            Column::CountTotal,
            Column::Text("•".to_owned()),
            Column::RemainingTime,
        ]
    );

    // Get tasks
    let mut out = TrackSetSet { sets: vec![], indicies: vec![] };
    for set in rx.into_iter() {
        out.sets.push(set);
        pb.update(1).ok();
    }

    Ok(out.verify()?)
}

/// Run optimizer on set of track sets
pub fn optimize_sets(set: TrackSetSet, tol: f32, max_iters: u64, extra_params: usize, rng_range: RangeInclusive<f32>) -> Option<HashMap<String, f32>> {
    // Generate mead params
    let param_count = set.indicies.len();
    let mut rng = rand::thread_rng();
    let inputs = (0..param_count + extra_params).into_iter().map(|_| (0..param_count).into_iter().map(|_| rng.gen_range(rng_range.clone())).collect()).collect();
    let opt: NelderMead<Vec<f32>, f32> = NelderMead::new(inputs).with_sd_tolerance(tol).unwrap();

    // Run optimizer
    info!("Indicies: {:?}", set.indicies);
    let mut res = Executor::new(set, opt)
        .configure(|state| state.max_iters(max_iters))
        .add_observer(Observer, ObserverMode::Always)
        .run()
        .unwrap();
    info!("Output: {}", res);
    let set = res.problem.take_problem().unwrap();
    info!("Indicies: {:?}", set.indicies);

    // Wrap
    if let Some(param) = res.state.best_param {
        return Some(HashMap::from_iter(set.indicies.into_iter().zip(param.into_iter())));
    }
    None
}


#[derive(Clone)]
pub struct TrackSet {
    /// file name: samples
    estimates: Vec<(String, Vec<f32>)>,
    reference: Vec<f32>,
}

impl TrackSet {
    /// Load estimates and reference tracks
    pub fn load(reference_path: impl AsRef<Path>, estimates_path: impl AsRef<Path>, filter: &[String], sample_rate: u32) -> Result<TrackSet, Error> {
        // Estimates
        let mut estimates = vec![];
        for entry in std::fs::read_dir(estimates_path)? {
            let entry = entry?;
            let name = entry.path().file_name().unwrap().to_string_lossy().to_string();
            if filter.is_empty() || filter.contains(&name) {
                let samples = decode_file(entry.path(), sample_rate)?;
                estimates.push((name, samples));
            }

        }
        // Reference
        let mut reference = decode_file(reference_path, sample_rate)?;
        // Uniform length
        let length = estimates.iter().map(|i| &i.1).chain([&reference]).map(|i| i.len()).min().unwrap();
        estimates.iter_mut().for_each(|i| i.1.truncate(length));
        reference.truncate(length);

        Ok(TrackSet { reference, estimates })
    }

    /// Create ensemble with weight
    pub fn ensemble_with_weights(&self, weights: &HashMap<String, f32>) -> Vec<f32> {
        let data = self.estimates.iter().filter_map(|(name, data)| match weights.get(name) {
            Some(w) => Some((data, *w)),
            None => None,
        }).collect::<Vec<_>>();
        ensemble(&data)
    }

    /// Calculate weights by SDR ratios
    /// Outputs: filename: weight
    pub fn weights_by_sdr_ratio(&self) -> HashMap<String, f32> {
        let mut sdrs = self.estimates
            .iter()
            .map(|(k, v)| (k.clone(), calculate_sdr_raw(&self.reference, v)))
            .collect::<HashMap<String, f32>>();
        let sum: f32 = sdrs.values().sum();
        sdrs.values_mut().for_each(|v| *v = *v / sum);
        sdrs
    }
}

/// Set of track sets
pub struct TrackSetSet {
    sets: Vec<TrackSet>,
    indicies: Vec<String>
}

impl CostFunction for TrackSetSet {
    type Param = Vec<f32>;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Create ensemble
        let sdrs = self.sets.par_iter().map(|i| {
            let data = i.estimates.iter()
                .map(|(name, data)| (data, param[self.indicies.iter().position(|i| i == name).unwrap()]))
                .collect::<Vec<_>>();
            let ensemble = ensemble(&data);
            let sdr = calculate_sdr_db(&i.reference, &ensemble);
            sdr
        }).collect::<Vec<_>>();
        let avg = sdrs.iter().fold(0.0, |acc, i| acc + *i) / sdrs.len() as f32;
        Ok(-avg)
    }
}

impl TrackSetSet {
    /// Verify this track set and set indicies
    pub fn verify(mut self) -> Result<TrackSetSet, Error> {
        if self.sets.is_empty() {
            return Err(anyhow!("No sets"));
        }
        self.indicies = self.sets[0].estimates.iter().map(|(n, _)| n.to_string()).collect();
        if !self.sets.iter().all(|s| s.estimates.iter().all(|e| self.indicies.contains(&e.0))) {
            return Err(anyhow!("Not all estimates are present for every set"));
        }
        Ok(self)
    }
}

/// Decode file to uniform format
pub fn decode_file(path: impl AsRef<Path>, sample_rate: u32) -> Result<Vec<f32>, Error> {
    let file = BufReader::new(File::open(path)?);
    let source = Decoder::new(file)?;
    let source: UniformSourceIterator<Decoder<BufReader<File>>, f32> = UniformSourceIterator::new(source, 2, sample_rate);
    Ok(source.collect())
}

/// Calculate SDR as dB
/// a = ref, b = est
pub fn calculate_sdr_db(a: &[f32], b: &[f32]) -> f32 {
    10.0 * (calculate_sdr_raw(a, b)).log10()
}

/// Calculate SDR (without converting to dB)
/// a = ref, b = est
/// https://stackoverflow.com/questions/72939521/how-to-calculate-metrics-sdr-si-sdr-sir-sar-in-python
pub fn calculate_sdr_raw(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    let num: f32 = a.iter().map(|s| s.powi(2)).sum::<f32>() + f32::EPSILON;
    let den: f32 = a.iter().zip(b).map(|(a, b)| (*a - *b).powi(2)).sum::<f32>() + f32::EPSILON;
    num / den
}

/// Ensemble inputs with weights
pub fn ensemble(inputs: &[(&Vec<f32>, f32)]) -> Vec<f32> {
    // Check lengths
    if inputs.len() == 0 {
        return vec![];
    }
    assert!(inputs.iter().all(|i| i.0.len() == inputs[0].0.len()));
    let len = inputs.len() as f32;
    // Allocate
    let mut output = vec![0f32; inputs[0].0.len()];
    
    // Merge
    for (samples, volume) in inputs {
        output.iter_mut().zip(samples.iter()).for_each(|(out, i)| *out += i * volume);
    }
    output.iter_mut().for_each(|i| *i /= len);
    output
}

/// Write wav to file for testing
pub fn write_wav(output: impl AsRef<Path>, data: &[f32], channels: u16, sample_rate: u32) -> Result<(), Error> {
    let mut writer = WavWriter::new(
        BufWriter::new(File::create(output)?), 
        WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    )?;
    for sample in data {
        writer.write_sample(*sample)?;
    }
    Ok(())
}

/// Calculate factorial
pub fn factorial(num: usize) -> usize {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

/// argmin logger
struct Observer;

impl<I> Observe<I> for Observer where I: State, I::Param: std::fmt::Debug {

    fn observe_init(&mut self, _name: &str, _kv: &argmin::core::KV) -> Result<(), Error> {
        Ok(())
    }

    fn observe_iter(&mut self, state: &I, _kv: &argmin::core::KV) -> Result<(), Error> {
        debug!("iter: {}, cost: {}, params: {:?}", state.get_iter(), state.get_cost(), state.get_best_param());
        Ok(())
    }
}
